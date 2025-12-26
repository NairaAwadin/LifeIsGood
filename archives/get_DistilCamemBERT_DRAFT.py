#!/usr/bin/env python3
"""
Train a fast text classifier to flag location contradictions in French real-estate listings.

Input CSV columns (required):
- postalCode (string; may be empty)
- city (string; may be empty)
- description (string; may be empty)
- label (0/1)

Output:
- Saves model + tokenizer to --out_dir
- Prints validation metrics
- Saves a small threshold report (optional)

How to run:
  python train_location_flagger.py --csv_path location_match_dataset_2000.csv --out_dir location_flagger

Tip:
- Use DistilCamemBERT for speed (default).
- Start with max_len=256 for latency; increase if needed.
"""

from __future__ import annotations

import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# -----------------------------
# 1) Reproducibility helpers
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# 2) Input building
# -----------------------------
def build_input_text(postal_code: str, city: str, description: str, desc_max_chars: int) -> str:
    """
    What: Build a single text string that contains all signals.
    Why: The model only sees "one sequence of tokens", so we must include structured fields.

    We keep preprocessing LIGHT to preserve cues:
    - Keep punctuation, accents, line breaks (often meaningful).
    - Truncate description to limit compute.
    """
    postal_code = "" if postal_code is None or pd.isna(postal_code) else str(postal_code)
    city = "" if city is None or pd.isna(city) else str(city)
    description = "" if description is None or pd.isna(description) else str(description)

    # light normalization (safe)
    description = description.replace("\r\n", "\n").replace("\r", "\n")
    if desc_max_chars > 0:
        description = description[:desc_max_chars]

    # "field tags" help the model separate columns
    return f"[POSTAL] {postal_code}\n[CITY] {city}\n[DESC]\n{description}"


# -----------------------------
# 3) Torch dataset
# -----------------------------
class TextClsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels.astype(int)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# -----------------------------
# 4) Metrics
# -----------------------------
def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    What: Compute quality on validation.
    Why: Accuracy alone can hide false-positive issues, so we track precision/recall/F1 too.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def threshold_report(
    probs: np.ndarray, labels: np.ndarray, thresholds: List[float] | None = None
) -> List[Dict[str, float]]:
    """
    What: Evaluate multiple thresholds on P(label=1).
    Why: You likely want high precision (avoid false positives), so threshold tuning matters.
    """
    if thresholds is None:
        thresholds = [x / 100 for x in range(10, 96, 5)]  # 0.10..0.95

    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        acc = accuracy_score(labels, preds)
        rows.append({"threshold": float(t), "accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)})
    return rows


# -----------------------------
# 5) Main train function
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True, help="CSV with postalCode, city, description, label")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to save the trained model")
    ap.add_argument("--model_name", type=str, default="cmarkea/distilcamembert-base", help="Base model checkpoint")
    ap.add_argument("--max_len", type=int, default=256, help="Tokenizer max_length (speed vs context)")
    ap.add_argument("--desc_max_chars", type=int, default=1800, help="Truncate raw description chars before tokenize")
    ap.add_argument("--test_size", type=float, default=0.15, help="Validation split size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train_bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    args = ap.parse_args()

    set_seed(args.seed)

    # --- Load data
    df = pd.read_csv(args.csv_path)
    required = {"postalCode", "city", "description", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure labels are valid
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not df["label"].isin([0, 1]).all():
        bad = df.loc[~df["label"].isin([0, 1]), "label"].unique().tolist()
        raise ValueError(f"Found non-binary labels: {bad}")

    # --- Build input text
    df["input_text"] = df.apply(
        lambda r: build_input_text(
            postal_code=r.get("postalCode", ""),
            city=r.get("city", ""),
            description=r.get("description", ""),
            desc_max_chars=args.desc_max_chars,
        ),
        axis=1,
    )

    # --- Train/val split (stratified keeps class balance)
    train_df, val_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["label"]
    )

    # --- Tokenizer + encode
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_enc = tokenizer(
        train_df["input_text"].tolist(),
        truncation=True,
        max_length=args.max_len,
    )
    val_enc = tokenizer(
        val_df["input_text"].tolist(),
        truncation=True,
        max_length=args.max_len,
    )

    train_ds = TextClsDataset(train_enc, train_df["label"].values)
    val_ds = TextClsDataset(val_enc, val_df["label"].values)

    # --- Model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # --- Training args
    use_fp16 = torch.cuda.is_available()
    train_args = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        fp16=use_fp16,
        report_to="none",
        logging_steps=50,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Train
    trainer.train()

    # --- Evaluate
    metrics = trainer.evaluate()
    print("\nFinal validation metrics:")
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"  {k:>12s}: {v:.4f}")
        else:
            print(f"  {k:>12s}: {v}")

    # --- Threshold tuning report on validation
    # Get probabilities for class 1 on val set
    pred_out = trainer.predict(val_ds)
    logits = pred_out.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    labels = val_df["label"].values

    t_rows = threshold_report(probs, labels)
    best = max(t_rows, key=lambda x: x["f1"])

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "threshold_report.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": t_rows, "best_by_f1": best}, f, ensure_ascii=False, indent=2)

    print("\nThreshold report saved to:", os.path.join(args.out_dir, "threshold_report.json"))
    print("Best threshold by F1:", best)

    # --- Save model + tokenizer
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("\nSaved model to:", args.out_dir)


if __name__ == "__main__":
    main()
