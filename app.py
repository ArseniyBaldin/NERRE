import os
import sys
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from collections import defaultdict

# Добавляем корень проекта в sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.docx_parser import parse_docx
from dataset.predict_ner import predict_entities
from transformers import AutoModelForTokenClassification
from training.train_utils import get_last_checkpoint
from config import label2id, id2label

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Загружаем модель один раз
model = AutoModelForTokenClassification.from_pretrained(
    get_last_checkpoint("training/models/ner_checkpoints")[0],
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

import html  # встроенный модуль

def highlight_text(text, entities):
    # Экранируем весь текст, чтобы < > & и т.п. не мешали HTML
    safe_text = html.escape(text)

    # Переносим координаты в экранированный текст
    shift_map = {}  # старый индекс → сдвиг в новом тексте
    original_index = 0
    new_index = 0
    for i, c in enumerate(text):
        escaped = html.escape(c)
        shift_map[i] = new_index
        new_index += len(escaped)

    tags = defaultdict(list)
    for ent in entities:
        start = shift_map.get(ent["start"], ent["start"])
        end = shift_map.get(ent["end"] - 1, ent["end"] - 1) + 1
        cls = ent["label"].lower()
        tags[start].append(f'<span class="entity {cls}">')
        tags[end].append('</span>')

    result = []
    for i, char in enumerate(safe_text):
        if i in tags:
            result.extend(tags[i])
        result.append(char)
    if len(safe_text) in tags:
        result.extend(tags[len(safe_text)])
    return ''.join(result)
from collections import defaultdict

def group_entities(entities):
    grouped = defaultdict(list)
    for ent in entities:
        grouped[ent["label"].upper()].append(ent["text"])
    return dict(sorted(grouped.items()))  # сортировка по типу сущности


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "text": "",
        "entities": [],
        "entity_types": [],
        "entities_grouped": {}  # <-- добавляем!
    })


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = parse_docx(file_path).replace("\n", " ")
    entities = predict_entities(text, model, mode='group')
    highlighted = highlight_text(text, entities)
    # Получаем связи
    relations, _ = predict_relations(text, entities, re_model, tokenizer, re_id2label)

    # Строим граф
    build_graph_html(relations, output_path="static/graph.html")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "text": highlighted,
        "entities_grouped": group_entities(entities),
        "entity_types": sorted({ent["label"].lower() for ent in entities})
    })

from transformers import AutoModelForSequenceClassification
from config import re_id2label, re_label2id, ALLOWED_RELATIONS, tokenizer
from dataset.predict_re import predict_relations  # или твой путь
from pyvis.network import Network

# Загружаем модель отношений один раз
re_model = AutoModelForSequenceClassification.from_pretrained(
    get_last_checkpoint("training/models/re_checkpoints")[0]
)

def build_graph_html(relations, output_path="static/graph.html"):
    net = Network(height="600px", width="100%", directed=True, notebook=False, cdn_resources="remote")
    net.barnes_hut()

    added = set()

    for rel in relations:
        subj = f"{rel['subj']['text']} ({rel['subj']['label']})"
        obj = f"{rel['obj']['text']} ({rel['obj']['label']})"
        label = rel['label']

        if subj not in added:
            net.add_node(subj, label=subj)
            added.add(subj)
        if obj not in added:
            net.add_node(obj, label=obj)
            added.add(obj)

        net.add_edge(subj, obj, label='.')

    net.set_options("""{
      "edges": {"arrows": {"to": {"enabled": true}}},
      "nodes": {"font": {"size": 16}},
      "physics": {"barnesHut": {"gravitationalConstant": -8000}}
    }""")

    net.write_html(output_path)
