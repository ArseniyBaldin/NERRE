import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from datasets import load_dataset
from config import re_label2id, re_id2label, DROPOUT, EPOCH, BATCH_SIZE, GRAD_ACCUM, tokenizer, MAX_LENGTH
import numpy as np
from sklearn.metrics import classification_report
from training.train_utils import get_last_checkpoint
from transformers import default_data_collator

# === Препроцессинг ===
def prepare_features(example):
    tokens = example["tokens"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    subj_mask = [0] * len(input_ids)
    obj_mask = [0] * len(input_ids)
    for i in range(*example["subj_span"]):
        subj_mask[i] = 1
    for i in range(*example["obj_span"]):
        obj_mask[i] = 1

    pad_len = MAX_LENGTH - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        subj_mask += [0] * pad_len
        obj_mask += [0] * pad_len
    else:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        subj_mask = subj_mask[:MAX_LENGTH]
        obj_mask = obj_mask[:MAX_LENGTH]
    # print(re_label2id[example["label"]])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "subj_mask": subj_mask,
        "obj_mask": obj_mask,
        "label": re_label2id[example["label"]]
    }

# === Метрики ===
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    print(classification_report(
        labels,
        preds,
        labels=list(re_id2label.keys()),
        target_names=[re_id2label[i] for i in list(re_id2label.keys())]
    ))
    return {
        "accuracy": (preds == labels).mean(),
        "report": classification_report(
            labels,
            preds,
            labels=list(re_id2label.keys()),  # 👈 добавляем список всех меток
            target_names=[re_id2label[i] for i in list(re_id2label.keys())],  #
            output_dict=True,
            zero_division=0  #
        )

    }

# === Обучение ===
from transformers import AutoModelForSequenceClassification

def train_re(epoch=EPOCH, batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM,
             output_dir="models/re_checkpoints", data_path="../dataset/re_dataset.jsonl"):

    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.train_test_split(test_size=0.01, seed=83, shuffle=True)
    train_ds = dataset["train"].map(prepare_features)
    val_ds = dataset["test"].map(prepare_features)

    last_ckpt, resume = get_last_checkpoint(output_dir)
    print(f'Загружена модель: {last_ckpt}')

    model = AutoModelForSequenceClassification.from_pretrained(
        get_last_checkpoint(output_dir)[0],
        num_labels=len(re_label2id),
        id2label=re_id2label,
        label2id=re_label2id
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.hidden_dropout_prob = DROPOUT
    model.config.attention_probs_dropout_prob = DROPOUT


    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        eval_accumulation_steps=grad_accum,
        num_train_epochs=epoch,
        weight_decay=0.01,
        overwrite_output_dir=True,
        save_total_limit=1,
        use_cpu=True,
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=train_ds,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    print(f'{"Продолжаем" if resume else "Начинаем"} обучение с {last_ckpt}')
    trainer.train(resume_from_checkpoint=resume)

if __name__ == "__main__":
    train_re()
