from transformers import (
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
from config import label2id, id2label, EPOCH, BATCH_SIZE, GRAD_ACCUM, DROPOUT
from config import tokenizer
from training.train_utils import get_last_checkpoint
from datasets import load_dataset
import torch
import numpy as np
from seqeval.metrics import classification_report, f1_score



def tokenize_and_align_labels(example, tokenizer):
    tokenized_inputs = tokenizer(example["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128)
    labels = []
    word_ids = tokenized_inputs.word_ids()

    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != prev_word_idx:
            labels.append(label2id.get(example["tags"][word_idx], 0))  # B-...
        else:
            labels.append(label2id.get(example["tags"][word_idx], 0))  # I-...
        prev_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 🧩 Батчевое маскирование
class DataCollatorWithRandomMasking(DataCollatorForTokenClassification):
    def __init__(self, tokenizer, label_pad_token_id=-100, mask_prob=0.2):
        super().__init__(tokenizer, label_pad_token_id=label_pad_token_id)
        self.mask_token_id = tokenizer.mask_token_id
        self.mask_prob = mask_prob

    def torch_call(self, features):
        batch = super().torch_call(features)
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Маскируем с вероятностью только валидные токены
        mask = (labels != -100) & (torch.rand_like(input_ids.float()) < self.mask_prob)
        input_ids[mask] = self.mask_token_id
        return batch

# 📊 Метрики с использованием seqeval
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    labels = p.label_ids

    true_preds = []
    true_labels = []

    for pred_seq, label_seq in zip(preds, labels):
        pred_tags = []
        label_tags = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:
                pred_tags.append(id2label[pred_id])
                label_tags.append(id2label[label_id])
        true_preds.append(pred_tags)
        true_labels.append(label_tags)
    print(classification_report(true_labels, true_preds))
    return {
        "f1": f1_score(true_labels, true_preds),
        "report": classification_report(true_labels, true_preds, output_dict=True)
    }

# 🚀 Обучение модели NER
def train_ner(epoch=EPOCH, batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM,
              output_dir="models/ner_checkpoints", data_path="../dataset/ner_bio_dataset.jsonl"):

    last_ckpt, resume = get_last_checkpoint(output_dir)
    print(f'Загружена модель: {last_ckpt}')
    # Загружаем весь датасет
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Разделяем вручную (например 90/10)
    dataset = dataset.train_test_split(test_size=0.01, seed=82, shuffle=True)
    train_dataset = dataset["train"].map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=False)
    test_dataset = dataset["test"].map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=False)

    # Загружаем модель с чекпоинта или модели по умолчанию
    model = AutoModelForTokenClassification.from_pretrained(
        last_ckpt,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model.config.hidden_dropout_prob = DROPOUT
    model.config.attention_probs_dropout_prob = DROPOUT
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # gradient_accumulation_steps=grad_accum,
        # eval_accumulation_steps=grad_accum,
        num_train_epochs=epoch,
        weight_decay=0.01,
        overwrite_output_dir=True,
        save_total_limit=1,
        use_cpu=True,  # оставляем как ты просил
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithRandomMasking(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    print(f'{"Продолжаем" if resume else "Начинаем"} обучение с {last_ckpt}')
    trainer.train(resume_from_checkpoint=resume)

if __name__ == "__main__":
    train_ner()
