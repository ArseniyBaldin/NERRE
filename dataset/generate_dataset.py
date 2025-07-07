from pathlib import Path
import json
import uuid
from typing import List, Literal, Tuple
import re

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from docx_parser import parse_docx
import os
from dotenv import load_dotenv

load_dotenv()
# === OpenAI setup ===
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4",
    openai_api_key=OPENAI_API_KEY
)

# === Entity schema ===
EntityLabel = Literal[
    "product", "property", "val", "amount",
    "client", "supplier", "address", "deadline", "action"
]

class Entity(BaseModel):
    start: int
    end: int
    text: str
    label: EntityLabel

# === Prompt template ===
template = """Вставь в исходный текст квадратные скобки вокруг всех сущностей и подпиши их тип через дефис.
Формат: [сущность - тип].

Типы сущностей:
- product — товар или услуга
- property — характеристика товара
- val — значение характеристики
- amount — количество
- client — заказчик
- supplier — поставщик
- address — место
- deadline — дедлайн
- action — то, к чему относится дедлайн

Размечай только значимые сущности. Верни полностью весь текст с разметкой.

Текст:
{text}
"""


prompt = PromptTemplate.from_template(template)

# === Split text ===
def split_text(text: str, window_size: int, stride: int) -> List[Tuple[str, int]]:
    windows = []
    i = 0
    while i < len(text):
        window = text[i:i+window_size]
        windows.append((window, i))
        if i + window_size >= len(text):
            break
        i += stride
    return windows

# === Parse GPT output in format [entity - label]
def parse_annotated_output(full_text: str, marked_text: str) -> List[Entity]:
    pattern = re.compile(r'\[(.+?) - (\w+?)\]')
    entities = []
    used_spans = set()

    for match in pattern.finditer(marked_text):
        ent_text, label = match.groups()
        label = label.lower()

        if label not in EntityLabel.__args__:
            continue

        # ищем непересекающееся вхождение
        start = full_text.find(ent_text)
        while start != -1 and any(start <= e <= start + len(ent_text) for e in used_spans):
            start = full_text.find(ent_text, start + 1)

        if start == -1:
            print(f"⚠️ Не найдено в тексте: {ent_text}")
            continue

        end = start + len(ent_text)
        used_spans.update(range(start, end))

        entities.append(Entity(
            start=start,
            end=end,
            text=ent_text,
            label=label
        ))

    return entities

# === Convert to Label Studio format ===
def to_labelstudio_format(text: str, entities: List[Entity]) -> dict:
    results = []
    for ent in entities:
        results.append({
            "value": {
                "start": ent.start,
                "end": ent.end,
                "text": ent.text,
                "labels": [ent.label]
            },
            "id": str(uuid.uuid4()),
            "from_name": "ner",
            "to_name": "text",
            "type": "labels",
            "origin": "model"
        })
    return {
        "data": {"text": text},
        "predictions": [{"result": results}]
    }

# === Main function ===
def main(text: str, window_size: int = 128, stride: int = 120, output_dir: str = "test_samples"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    windows = split_text(text, window_size, stride)
    print(f"🔍 Разделено на {len(windows)} окон")

    all_entities: List[Entity] = []

    for i, (window_text, offset) in enumerate(windows):
        print(f"🧠 GPT размечает окно {i+1}/{len(windows)}...")
        prompt_text = prompt.format(text=window_text.replace("\n", " "))
        response = llm.invoke(prompt_text)
        marked_text = response.content

        try:
            entities = parse_annotated_output(text, marked_text)
            all_entities.extend(entities)
        except Exception as e:
            print(f"⚠️ Ошибка при обработке окна {i+1}: {e}")

    # удаляем дубликаты по (start, end, label)
    unique_entities = {
        (e.start, e.end, e.label): e for e in all_entities
    }.values()

    full_result = to_labelstudio_format(text, list(unique_entities))
    out_file = output_path / "full_sample.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)

    print(f"✅ Финальный файл сохранён: {out_file.absolute()}")

# main(text='''Поставщик — Общество с ограниченной ответственностью «ЭкоСнаб», обязуется поставить товар — 3D наклейки с глянцевым покрытием (размер 10x15 см), в количестве 500 штук, по адресу: г. Казань, ул. Академика Глушко, д. 22.
#
# Поставка должна быть осуществлена не позднее 10 августа 2025 года. Заказчик — ООО «Рекламные Решения».
#
# Согласно условиям договора, передача товара осуществляется по предварительной заявке не ранее 3 недель после подписания контракта.
# '''.replace("\n", " "))

main(text=parse_docx('docx_files/Техническое задание.docx').replace("\n", " "))