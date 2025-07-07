import torch
from config import (
    ALLOWED_RELATIONS, RE_ENTITY_TOTAL_SPAN, RE_WINDOW_SIZE,
    RE_MIDDLE_MAX_TOKENS, re_label2id, re_id2label, tokenizer
)
from dataset.prepare_re import build_context
from training.train_utils import get_last_checkpoint
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from dataset.predict_ner import predict_entities
from dataset.docx_parser import parse_docx
import networkx as nx


def predict_relations(text, entities, model, tokenizer, id2label, device="cpu"):
    model.eval()
    model.to(device)
    relations = []

    G = nx.DiGraph()  # ← граф создаётся здесь

    for i, ent1 in enumerate(entities):
        for j, ent2 in enumerate(entities):
            if i == j:
                continue

            ent1["label"] = ent1["label"].upper()
            ent2["label"] = ent2["label"].upper()
            #
            if ((ent1["label"], ent2["label"]) and (ent2["label"], ent1["label"])) not in ALLOWED_RELATIONS:
                continue

            # if ((ent1["label"], ent2["label"])) not in ALLOWED_RELATIONS:
            #     continue

            result = build_context(text, ent1, ent2)
            if result is None:
                continue

            context_str, tokens, (s1, e1), (s2, e2) = result

            inputs = tokenizer(
                context_str,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                pred_label_id = torch.argmax(outputs.logits, dim=1).item()

            pred_label = id2label[pred_label_id]
            if pred_label != "no_relation":
                subj_text = tokens[s1:e1]
                obj_text = tokens[s2:e2]

                print("📜 Контекст:")
                print(context_str)
                print(f"🟨 Субъект [{ent1['label']}]: {' '.join(subj_text)}")
                print(f"🟦 Объект  [{ent2['label']}]: {' '.join(obj_text)}")
                print(f"🔗 Предсказанная связь: {pred_label}")
                print("-" * 80)

                relations.append({
                    "subj": {"text": ent1["text"], "label": ent1["label"]},
                    "obj": {"text": ent2["text"], "label": ent2["label"]},
                    "label": pred_label
                })

                # === Добавление в граф ===
                subj_node = f"{ent1['text']} ({ent1['label']})"
                obj_node = f"{ent2['text']} ({ent2['label']})"
                G.add_node(subj_node)
                G.add_node(obj_node)
                G.add_edge(subj_node, obj_node, label=pred_label)

    return relations, G

def run_relation_prediction(docx_path, re_checkpoint_path, ner_checkpoint_path=None):
    text = parse_docx(docx_path).replace("\n", " ")

    if ner_checkpoint_path is None:
        ner_checkpoint_path = get_last_checkpoint("../training/models/ner_checkpoints")[0]
    if re_checkpoint_path is None:
        re_checkpoint_path = get_last_checkpoint("../training/models/re_checkpoints")[0]

    ner_model = AutoModelForTokenClassification.from_pretrained(ner_checkpoint_path)
    entities = predict_entities(text, ner_model)

    re_model = AutoModelForSequenceClassification.from_pretrained(re_checkpoint_path)
    re_model.eval()

    relations, graph = predict_relations(text, entities, re_model, tokenizer, re_id2label)
    return relations, graph

if __name__ == "__main__":
    rels, G = run_relation_prediction(
        docx_path="docx_files/Техническое задание.docx",
        re_checkpoint_path="../training/models/re_checkpoints/checkpoint-126"
    )
    print(rels)

    # Пример визуализации
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'label')

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.tight_layout()
    plt.show()
