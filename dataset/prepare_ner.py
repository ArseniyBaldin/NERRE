import json
import re
import uuid
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from datasets import Dataset
from config import tokenizer, SPECIAL_TOKENS

# === Настройки ===
MAX_TOKENS = 128
STRIDE = 64


def get_chunks_with_offsets(text: str):
    """
    Возвращает список:
    (chunk_text, is_table, global_start)
    """
    chunks = []
    table_pattern = re.compile(r"\[TABLE](.*?)\[/TABLE]", re.DOTALL)
    last_idx = 0

    for match in table_pattern.finditer(text):
        start, end = match.span()
        before = text[last_idx:start]

        for sent in sent_tokenize(before):
            sent = sent.strip()
            if sent:
                idx = text.find(sent, last_idx)
                chunks.append((sent, False, idx))

        table_body = match.group(1)
        for row_match in re.finditer(r"\[ROW](.*?)[/]ROW]", match.group(0), re.DOTALL):
            full_row_text = row_match.group(0)
            row_content = row_match.group(1).strip()
            global_start = text.find(full_row_text, last_idx)
            if global_start != -1:
                content_start = global_start + full_row_text.find(row_content)
                chunks.append((row_content, True, content_start))

        last_idx = end

    # После таблиц
    if last_idx < len(text):
        after = text[last_idx:]
        for sent in sent_tokenize(after):
            sent = sent.strip()
            if sent:
                idx = text.find(sent, last_idx)
                chunks.append((sent, False, idx))

    return chunks

bio_samples = []  # глобальный список для накопления

def print_chunk_info(text_chunk, global_start, entities, is_table):
    label = "[TABLE]" if is_table else "[TEXT]"

    # Используем все сущности — отфильтруем по окну позже
    active_entities = [
        {
            "start": ent["start"],
            "end": ent["end"],
            "label": ent["label"],
            "text": ent["text"]
        }
        for ent in entities
    ]

    encoding = tokenizer(
        text_chunk,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

    n_tokens = len(tokens)
    windows = []

    if n_tokens <= MAX_TOKENS:
        windows = [(0, n_tokens)]
    else:
        for start in range(0, n_tokens, STRIDE):
            end = start + MAX_TOKENS
            windows.append((start, min(end, n_tokens)))
            if end >= n_tokens:
                break

    for w_idx, (start_i, end_i) in enumerate(windows):
        sub_tokens = tokens[start_i:end_i]
        sub_offsets = offsets[start_i:end_i]
        bio_tags = ["O"] * len(sub_tokens)

        # Абсолютные координаты окна
        abs_window_start = global_start + sub_offsets[0][0]
        abs_window_end = global_start + sub_offsets[-1][1]

        # Реальный текст окна
        start_char = sub_offsets[0][0]
        end_char = sub_offsets[-1][1]
        window_text = text_chunk[start_char:end_char]

        print(f"\n--- {label} | window {w_idx} | токены {end_i - start_i} ---")
        print(window_text)

        print("Сущности:")
        printed_ids = set()
        for ent in active_entities:
            ent_abs_start = ent["start"]
            ent_abs_end = ent["end"]

            if ent_abs_end <= abs_window_start or ent_abs_start >= abs_window_end:
                continue

            first = True
            for i, (start_off, end_off) in enumerate(sub_offsets):
                token_abs_start = global_start + start_off
                token_abs_end = global_start + end_off

                if token_abs_end <= ent_abs_start or token_abs_start >= ent_abs_end:
                    continue

                bio_tags[i] = f"B-{ent['label']}" if first else f"I-{ent['label']}"
                first = False

            if ent_abs_start not in printed_ids:
                local_start = ent_abs_start - abs_window_start
                local_end = ent_abs_end - abs_window_start
                print(f"  {ent['label']:>10} | {ent['text']} ({ent_abs_start}–{ent_abs_end}) [лок: {local_start}–{local_end}]")
                printed_ids.add(ent_abs_start)

        print("\nBIO-токены:")
        print("TOKENS: ", " ".join(sub_tokens))
        print("TAGS:   ", " ".join(bio_tags))
        print("-" * 40)
        bio_samples.append({
            "tokens": sub_tokens,
            "tags": bio_tags
        })


def iterate_labelstudio_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        print(f"\n=================== СЕМПЛ {i} ===================")
        text = item["data"]["text"]
        results = item["annotations"][0]["result"]

        entities = [
            {
                "start": res["value"]["start"],
                "end": res["value"]["end"],
                "label": res["value"]["labels"][0],
                "text": res["value"]["text"]
            }
            for res in results
            if res["type"] == "labels" and res["origin"] in {"manual", "prediction-changed"}
        ]

        for chunk_text, is_table, global_start in get_chunks_with_offsets(text):
            print_chunk_info(chunk_text, global_start, entities, is_table)

if __name__ == "__main__":
    iterate_labelstudio_json("from_ls/labeled.json")
    import json

    with open("ner_bio_dataset.jsonl", "w", encoding="utf-8") as f:
        for sample in bio_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")