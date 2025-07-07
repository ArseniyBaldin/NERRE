import json
import random
from config import tokenizer, RE_MIDDLE_MAX_TOKENS, RE_ENTITY_TOTAL_SPAN, RE_WINDOW_SIZE, NUM_RANDOM_NEGATIVE, ALLOWED_RELATIONS, label_mapping



def find_token_span(offsets, start, end):
    tok_start, tok_end = None, None
    for i, (s, e) in enumerate(offsets):
        if s <= start < e or (start <= s < end):
            if tok_start is None:
                tok_start = i
            if s < end:
                tok_end = i + 1
    return tok_start, tok_end

def build_context(text, ent1, ent2):
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

    s1, e1 = find_token_span(offsets, ent1["start"], ent1["end"])
    s2, e2 = find_token_span(offsets, ent2["start"], ent2["end"])

    if None in (s1, e1, s2, e2):
        # print("None in (s1, e1, s2, e2)")
        return None
    if s1 > s2:
        s1, e1, ent1, s2, e2, ent2 = s2, e2, ent2, s1, e1, ent1
    if (e2 - s1) > RE_ENTITY_TOTAL_SPAN:
        # print("(e2 - s1) > RE_ENTITY_TOTAL_SPAN")
        return None

    subj = [f"[SUBJ:{ent1['label']}]", *tokens[s1:e1], "[/SUBJ]"]
    obj = [f"[OBJ:{ent2['label']}]", *tokens[s2:e2], "[/OBJ]"]
    middle = tokens[e1:s2]
    middle_section = subj + middle + obj

    remaining = RE_MIDDLE_MAX_TOKENS - len(middle_section)
    if remaining < 0:
        return None

    half_window = min(RE_WINDOW_SIZE, remaining // 2)
    left = max(0, s1 - half_window)
    right = min(len(tokens), e2 + half_window)

    left_context = tokens[left:s1]
    right_context = tokens[e2:right]

    final_tokens = left_context + middle_section + right_context

    subj_start = len(left_context) + 1
    subj_end = subj_start + (e1 - s1)
    obj_start = subj_end + len(middle) + 2
    obj_end = obj_start + (e2 - s2)

    return tokenizer.convert_tokens_to_string(final_tokens), final_tokens, (subj_start, subj_end), (obj_start, obj_end)

def get_entities_and_relations(results):
    entities = {}
    relations = []
    for item in results:
        if item["type"] == "labels":
            eid = item["id"]
            label = item["value"]["labels"][0].upper()
            entities[eid] = {
                "start": item["value"]["start"],
                "end": item["value"]["end"],
                "text": item["value"]["text"],
                "label": label
            }
        elif item["type"] == "relation":
            relations.append((item["from_id"], item["to_id"]))
    return entities, relations

def process_labelstudio_json(path, output_path='re_dataset.jsonl'):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []

    for task in data:
        text = task["data"]["text"]
        results = task["annotations"][0]["result"]
        entities, relations = get_entities_and_relations(results)
        existing_pairs = set(relations)

        # === Положительные примеры ===
        for from_id, to_id in existing_pairs:
            if from_id not in entities or to_id not in entities:
                continue
            ent1, ent2 = entities[from_id], entities[to_id]

            # Разворачиваем
            if (ent1["label"], ent2["label"]) not in ALLOWED_RELATIONS and (ent2["label"], ent1["label"]) in ALLOWED_RELATIONS:
                ent1, ent2 = ent2, ent1

            if (ent1["label"], ent2["label"]) not in ALLOWED_RELATIONS:
                continue

            result = build_context(text, ent1, ent2)
            if result is None:
                continue

            context_str, token_list, (s1, e1), (s2, e2) = result
            relation_class = label_mapping.get(ent2["label"], "no_relation")

            samples.append({
                "text": context_str,
                "tokens": token_list,
                "label": relation_class,
                "subj_type": ent1["label"],
                "obj_type": ent2["label"],
                "subj_span": [s1, e1],
                "obj_span": [s2, e2]
            })

        # === Отрицательные примеры ===
        entity_list = list(entities.items())
        for i, (eid1, ent1) in enumerate(entity_list):
            # Формируем список допустимых, но не размеченных пар
            candidates = [
                (eid2, ent2) for j, (eid2, ent2) in enumerate(entity_list)
                if eid1 != eid2
                and (eid1, eid2) not in existing_pairs
                and (ent1["label"], ent2["label"]) in ALLOWED_RELATIONS
            ]
            random.shuffle(candidates)
            count = 0
            for eid2, ent2 in candidates:
                result = build_context(text, ent1, ent2)
                if result is None:
                    continue
                context_str, token_list, (s1, e1), (s2, e2) = result

                print(f"❌ Добавляем no_relation: {ent1['text']} ({ent1['label']}) → {ent2['text']} ({ent2['label']})")

                samples.append({
                    "text": context_str,
                    "tokens": token_list,
                    "label": "no_relation",
                    "subj_type": ent1["label"],
                    "obj_type": ent2["label"],
                    "subj_span": [s1, e1],
                    "obj_span": [s2, e2]
                })
                count += 1
                if count >= NUM_RANDOM_NEGATIVE:
                    break

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

# Запуск:
if __name__ =="__main__":
    process_labelstudio_json("from_ls/labeled.json")
