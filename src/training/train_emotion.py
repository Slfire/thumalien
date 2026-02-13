"""Entraînement du modèle d'analyse émotionnelle.

Usage:
    python -m src.training.train_emotion [--epochs 3] [--batch-size 8]
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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    EMOTION_DATA_DIR,
    EMOTION_MODEL_DIR,
    EVAL_BATCH_SIZE,
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
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}


def load_emotion_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge le dataset émotion (CSV ou TXT avec séparateur ;)."""
    csv_names = {
        "train": ["training.csv", "train.csv"],
        "val": ["validation.csv", "val.csv"],
        "test": ["test.csv"],
    }
    txt_names = {
        "train": ["train.txt"],
        "val": ["val.txt"],
        "test": ["test.txt"],
    }

    dfs = {}
    for split, filenames in csv_names.items():
        for fname in filenames:
            path = os.path.join(data_dir, fname)
            if os.path.exists(path):
                dfs[split] = pd.read_csv(path)
                logger.info("Chargé {} : {} lignes", path, len(dfs[split]))
                break

    if len(dfs) < 3:
        for split, filenames in txt_names.items():
            if split in dfs:
                continue
            for fname in filenames:
                path = os.path.join(data_dir, fname)
                if os.path.exists(path):
                    dfs[split] = pd.read_csv(
                        path, sep=";", header=None, names=["text", "label"]
                    )
                    logger.info("Chargé {} (txt) : {} lignes", path, len(dfs[split]))
                    break

    if len(dfs) < 3:
        raise FileNotFoundError(
            f"Dataset émotion incomplet dans {data_dir}. "
            f"Splits trouvés : {list(dfs.keys())}. "
            "Téléchargez depuis Kaggle : parulpandey/emotion-dataset"
        )

    for split in dfs:
        df = dfs[split]
        if df["label"].dtype == object:
            df["label"] = df["label"].map(LABEL2ID)
        dfs[split] = df.dropna(subset=["text", "label"])
        dfs[split]["label"] = dfs[split]["label"].astype(int)

    return dfs["train"], dfs["val"], dfs["test"]


def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Tokenise un DataFrame en HuggingFace Dataset."""
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
    """Calcule accuracy et F1 macro."""
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--data-dir", type=str, default=EMOTION_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=EMOTION_MODEL_DIR)
    args = parser.parse_args()

    device = get_device()

    # 1. Load data
    df_train, df_val, df_test = load_emotion_data(args.data_dir)
    logger.info(
        "Splits : {} train, {} val, {} test",
        len(df_train),
        len(df_val),
        len(df_test),
    )

    # 2. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset = tokenize_dataset(df_train, tokenizer)
    val_dataset = tokenize_dataset(df_val, tokenizer)
    test_dataset = tokenize_dataset(df_test, tokenizer)

    # 3. Model + LoRA
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(EMOTION_LABELS),
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

    # 4. Training
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
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        gradient_accumulation_steps=4,
        fp16=False,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Début de l'entraînement émotion...")
    monitor = EnergyMonitor()
    monitor.start()
    trainer.train()
    energy = monitor.stop()
    energy["task"] = "training_emotion"
    from datetime import datetime, timezone
    energy["timestamp"] = datetime.now(timezone.utc).isoformat()
    EnergyMonitor.save_record(energy)

    # 5. Evaluation sur test set
    test_results = trainer.evaluate(test_dataset)
    logger.info("Résultats test : {}", test_results)

    preds_output = trainer.predict(test_dataset)
    preds = preds_output.predictions.argmax(axis=-1)
    labels = preds_output.label_ids

    print("\n=== Classification Report (Test Set) ===")
    print(classification_report(labels, preds, target_names=EMOTION_LABELS))

    # 6. Save
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Modèle sauvegardé dans {}", args.output_dir)

    # 7. Metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    metrics = {
        "accuracy": float(test_results.get("eval_accuracy", 0)),
        "f1_macro": float(test_results.get("eval_f1_macro", 0)),
        "base_model": BASE_MODEL,
        "num_classes": len(EMOTION_LABELS),
        "labels": EMOTION_LABELS,
        "epochs": args.epochs,
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Métriques sauvegardées dans {}", metrics_path)


if __name__ == "__main__":
    main()
