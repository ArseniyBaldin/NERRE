import torch
from transformers import AutoModelForTokenClassification
from collections import defaultdict
from config import tokenizer, id2label, label2id
from dataset.prepare_ner import get_chunks_with_offsets  # если ты сохранишь его в отдельном модуле
from training.train_utils import get_last_checkpoint
MAX_TOKENS = 128
STRIDE = 64


# import torch
# from transformers import AutoModelForTokenClassification
# from config import tokenizer, id2label
# from your_chunking_module import get_chunks_with_offsets  # замени на свой путь

MAX_TOKENS = 128
STRIDE = 64


def bio_decode_old(predicted, tokens, offsets, global_start, original_text):
    entities = []
    current_entity = None

    for tag, token, (start_off, end_off) in zip(predicted, tokens, offsets):
        start_char = start_off.item()
        end_char = end_off.item()

        if tag.startswith("B-"):
            if current_entity:
                current_entity["text"] = original_text[current_entity["start"]:current_entity["end"]]
                entities.append(current_entity)
            current_entity = {
                "label": tag[2:],
                "start": global_start + start_char,
                "end": global_start + end_char
            }
        elif tag.startswith("I-") and current_entity and tag[2:] == current_entity["label"]:
            current_entity["end"] = global_start + end_char
        else:
            if current_entity:
                current_entity["text"] = original_text[current_entity["start"]:current_entity["end"]]
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        current_entity["text"] = original_text[current_entity["start"]:current_entity["end"]]
        entities.append(current_entity)

    return entities

def bio_decode(predicted, tokens, offsets, global_start, original_text):
    """
    Стандартная BIO-декодировка: объединяет подряд идущие теги одного типа
    """
    entities = []
    current_entity = None

    for tag, token, (start_off, end_off) in zip(predicted, tokens, offsets):
        start_char = start_off.item()
        end_char = end_off.item()

        if tag == "O":
            if current_entity:
                current_entity["text"] = original_text[current_entity["start"]:current_entity["end"]]
                entities.append(current_entity)
                current_entity = None
            continue

        label = tag.replace("B-", "").replace("I-", "")

        if current_entity and label == current_entity["label"]:
            current_entity["end"] = global_start + end_char
        else:
            if current_entity:
                current_entity["text"] = original_text[current_entity["start"]:current_entity["end"]]
                entities.append(current_entity)
            current_entity = {
                "label": label,
                "start": global_start + start_char,
                "end": global_start + end_char
            }

    if current_entity:
        current_entity["text"] = original_text[current_entity["start"]:current_entity["end"]]
        entities.append(current_entity)

    return entities


def bio_decode_split(predicted, tokens, offsets, global_start, original_text):
    """
    Альтернативный режим: каждая метка (B или I) — отдельная сущность
    """
    entities = []

    for tag, token, (start_off, end_off) in zip(predicted, tokens, offsets):
        if tag == "O":
            continue

        start_char = start_off.item()
        end_char = end_off.item()
        label = tag.replace("B-", "").replace("I-", "")
        start = global_start + start_char
        end = global_start + end_char

        entities.append({
            "label": label,
            "start": start,
            "end": end,
            "text": original_text[start:end]
        })

    return entities


def predict_entities(text: str, model, device="cpu", mode="group"):
    model.eval()
    model.to(device)
    all_entities = []

    decode_fn = bio_decode if mode == "group" else bio_decode_split

    for chunk_text, is_table, global_start in get_chunks_with_offsets(text):
        encoding = tokenizer(
            chunk_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False
        )

        input_ids = encoding["input_ids"][0]
        offsets = encoding["offset_mapping"][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        n_tokens = len(tokens)
        windows = []
        if n_tokens <= MAX_TOKENS:
            windows = [(0, n_tokens)]
        else:
            for start in range(0, n_tokens, STRIDE):
                end = min(start + MAX_TOKENS, n_tokens)
                windows.append((start, end))
                if end == n_tokens:
                    break

        for start_i, end_i in windows:
            chunk_ids = input_ids[start_i:end_i].unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_ids=chunk_ids)
                predictions = outputs.logits.argmax(dim=-1).squeeze(0).tolist()

            tokens_sub = tokens[start_i:end_i]
            offsets_sub = offsets[start_i:end_i]
            tags = [id2label[p] for p in predictions]

            entities = decode_fn(tags, tokens_sub, offsets_sub, global_start, text)
            all_entities.extend(entities)

    return merge_close_entities(all_entities)


def merge_entities(entities):
    """
    Убирает дубликаты и вложенные сущности
    """
    seen = set()
    merged = []
    for ent in entities:
        key = (ent["start"], ent["end"], ent["label"])
        if key not in seen:
            seen.add(key)
            merged.append(ent)
    return merged
def merge_close_entities(entities, max_gap=40):
    """
    Объединяет сущности одного типа, если они расположены близко (≤ max_gap символов между ними)
    """
    if not entities:
        return []

    entities = sorted(entities, key=lambda e: e["start"])
    merged = []
    current = entities[0]

    for ent in entities[1:]:
        if ent["label"] == current["label"] and ent["start"] - current["end"] <= max_gap:
            # объединяем
            current["end"] = ent["end"]
            current["text"] = current["text"] + ' ' + ent["text"]  # можно аккуратнее с пробелами
        else:
            merged.append(current)
            current = ent

    merged.append(current)
    return merged


if __name__ == "__main__":

    # from config import model_path
    from dataset.docx_parser import parse_docx
    # model = AutoModelForTokenClassification.from_pretrained(
    #     get_last_checkpoint("../training/models/ner_checkpoints")[0],
    #     num_labels=len(label2id),
    #     id2label=id2label,
    #     label2id=label2id
    # )
    model = AutoModelForTokenClassification.from_pretrained(get_last_checkpoint("../training/models/ner_checkpoints")[0])

    text = parse_docx("docx_files/приложение ТФД поставки до 100 000 руб.docx").replace("\n", " ")

    entities = predict_entities(text, model)
    for ent in entities:
        print(f"{ent['text']}  [{ent['label']}] ({ent['start']}–{ent['end']})")
