from transformers import AutoTokenizer

# 🔧 Модель и токенизатор
BASE_MODEL = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
MAX_LENGTH = 128

# 📌 BIO-разметка (NER)
label2id = {
    'O': 0,
    'B-action': 1, 'I-action': 2,
    'B-address': 3, 'I-address': 4,
    'B-amount': 5, 'I-amount': 6,
    'B-client': 7, 'I-client': 8,
    'B-deadline': 9, 'I-deadline': 10,
    'B-product': 11, 'I-product': 12,
    'B-property': 13, 'I-property': 14,
    'B-supplier': 15, 'I-supplier': 16,
    'B-date': 17, 'I-date': 18,
    'B-deadline_object': 19, 'I-deadline_object': 20,
    'B-responsibility': 21, 'I-responsibility': 22,
    'B-responsibility_object': 23, 'I-responsibility_object': 24,
    'B-property_num': 25, "I-property_num": 26,
    'B-requirements': 27, "I-requirements": 28,
    'B-cost': 29, "I-cost": 30,
    'B-val': 31, "I-val": 32
}
id2label = {v: k for k, v in label2id.items()}

# 📍 Уникальные типы сущностей
entity_types = sorted({label.split("-")[1] for label in label2id if "-" in label})

# 🏷️ Типы отношений (RE)
RE_LABELS = [
    "no_relation",
    "deadline_of",
    "action_of",
    "address_of",
    "responsibility_of",
    "date_of",
    "amount_of",
    "property_of",
    "property_num_of",
    "val_of"
]
re_label2id = {label: i for i, label in enumerate(RE_LABELS)}
re_id2label = {i: label for label, i in re_label2id.items()}

# 🔖 Спецтокены
re_tokens = [f"[SUBJ:{t.upper()}]" for t in entity_types] + \
            [f"[OBJ:{t.upper()}]" for t in entity_types] + ["[/SUBJ]", "[/OBJ]"]
table_tokens = (
    ["[TABLE]", "[/TABLE]", "[ROW]", "[/ROW]"] +
    [f"[COL{i}]" for i in range(1, 10)] +
    [f"[/COL{i}]" for i in range(1, 10)]
)
SPECIAL_TOKENS = re_tokens + table_tokens
tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

# ⚙️ RE параметры
RE_WINDOW_SIZE = 16                 # боковое окно
RE_ENTITY_TOTAL_SPAN = 128       # максимум расстояние между сущностями (в токенах)
RE_MIDDLE_MAX_TOKENS = 128         # максимум токенов внутри + маски
NUM_RANDOM_NEGATIVE = 3

# 🎯 Параметры обучения
EPOCH = 100
BATCH_SIZE = 16
GRAD_ACCUM = 4
DROPOUT = 0.3

ALLOWED_RELATIONS = {
    ("ACTION", "DEADLINE"),
    ("CLIENT", "ACTION"),
    ("CLIENT", "ADDRESS"),
    ("CLIENT", "RESPONSIBILITY"),
    ("DEADLINE_OBJECT", "ACTION"),
    ("DEADLINE_OBJECT", "DATE"),
    ("DEADLINE_OBJECT", "DEADLINE"),
    ("PRODUCT", "AMOUNT"),
    ("PRODUCT", "PROPERTY"),
    ("PRODUCT", "PROPERTY_NUM"),
    ("PRODUCT", "RESPONSIBILITY"),
    ("PROPERTY", "ACTION"),
    ("PROPERTY", "AMOUNT"),
    ("PROPERTY", "RESPONSIBILITY"),
    ("PROPERTY", "VAL"),
    ("RESPONSIBILITY", "DEADLINE"),
    ("RESPONSIBILITY_OBJECT", "ACTION"),
    ("RESPONSIBILITY_OBJECT", "RESPONSIBILITY"),
    ("SUPPLIER", "ACTION"),
    ("SUPPLIER", "RESPONSIBILITY")
}

label_mapping = {
    "DEADLINE": "deadline_of",
    "ACTION": "action_of",
    "ADDRESS": "address_of",
    "RESPONSIBILITY": "responsibility_of",
    "DATE": "date_of",
    "AMOUNT": "amount_of",
    "PROPERTY": "property_of",
    "PROPERTY_NUM": "property_num_of",
    "VAL": "val_of"
}