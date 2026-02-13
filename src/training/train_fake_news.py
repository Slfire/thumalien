"""Entraînement du modèle de détection fake news.

Usage:
    python -m src.training.train_fake_news [--epochs 3] [--batch-size 8]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from loguru import logger
from peft import TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    EVAL_BATCH_SIZE,
    FAKE_NEWS_DATA_DIR,
    FAKE_NEWS_MODEL_DIR,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
)
from src.energy.monitor import EnergyMonitor
from src.training.utils import get_device, get_lora_config

BASE_MODEL = "distilbert-base-uncased"
LABEL2ID = {"REAL": 0, "FAKE": 1}
ID2LABEL = {0: "REAL", 1: "FAKE"}


def load_data(data_dir: str) -> pd.DataFrame:
    """Charge et combine True.csv et Fake.csv."""
    true_path = os.path.join(data_dir, "True.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(
            f"True.csv et/ou Fake.csv introuvables dans {data_dir}. "
            "Téléchargez depuis Kaggle : clmentbisaillon/fake-and-real-news-dataset"
        )

    df_true = pd.read_csv(true_path)
    df_true["label"] = 0

    df_fake = pd.read_csv(fake_path)
    df_fake["label"] = 1

    df = pd.concat([df_true, df_fake], ignore_index=True)
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["text", "label"]].dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        "Dataset chargé : {} exemples ({} real, {} fake)",
        len(df),
        len(df_true),
        len(df_fake),
    )
    return df


def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Tokenise le DataFrame en HuggingFace Dataset."""
    dataset = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    return dataset


def compute_metrics(eval_pred):
    """Calcule accuracy et F1 pour le Trainer."""
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train fake news classifier")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--data-dir", type=str, default=FAKE_NEWS_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=FAKE_NEWS_MODEL_DIR)
    args = parser.parse_args()

    device = get_device()

    # 1. Load data
    df = load_data(args.data_dir)

    # 2. Train/val split 80/20
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]
    logger.info("Split : {} train, {} val", len(df_train), len(df_val))

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset = tokenize_dataset(df_train, tokenizer)
    val_dataset = tokenize_dataset(df_val, tokenizer)

    # 4. Model + LoRA
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    lora_config = get_lora_config(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        gradient_accumulation_steps=4,
        fp16=False,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=0,
    )

    # 6. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Début de l'entraînement fake news...")
    monitor = EnergyMonitor()
    monitor.start()
    trainer.train()
    energy = monitor.stop()
    energy["task"] = "training_fake_news"
    from datetime import datetime, timezone
    energy["timestamp"] = datetime.now(timezone.utc).isoformat()
    EnergyMonitor.save_record(energy)

    # 7. Evaluation finale
    eval_results = trainer.evaluate()
    logger.info("Résultats évaluation : {}", eval_results)

    preds_output = trainer.predict(val_dataset)
    preds = preds_output.predictions.argmax(axis=-1)
    labels = preds_output.label_ids

    print("\n=== Classification Report ===")
    print(classification_report(labels, preds, target_names=["REAL", "FAKE"]))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(labels, preds))

    # 8. Save
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Modèle sauvegardé dans {}", args.output_dir)

    # 9. Metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    metrics = {
        "accuracy": float(eval_results.get("eval_accuracy", 0)),
        "f1": float(eval_results.get("eval_f1", 0)),
        "base_model": BASE_MODEL,
        "epochs": args.epochs,
        "train_size": len(df_train),
        "val_size": len(df_val),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Métriques sauvegardées dans {}", metrics_path)


if __name__ == "__main__":
    main()
