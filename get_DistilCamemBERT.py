"""
PIPELINE :

1) Read CSV with postalCode, city, description, label; validate labels are 0/1.
2) Build input text per row with tags:
    [POSTAL] ...
    [CITY] ...
    [DESC]
    description...
    (optionally truncate description).
3) Stratified train/val split.
4) Tokenize with the chosen model (--model_name, default cmarkea/distilcamembert-base), using max_len.
5) Create torch datasets from tokenized inputs + labels.
6) Load AutoModelForSequenceClassification (2 labels).
7) Configure TrainingArguments (epochs, LR, batch sizes, eval/save each epoch, fp16 if CUDA).
8) Train with Trainer (dynamic padding via DataCollatorWithPadding).
09) Save the model and tokenizer to --out_dir.
"""
from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
"""
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F1 = 2PR/(P+R) where P is precision and R is recall
"""
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def threshold_report(
    probs: np.ndarray, labels: np.ndarray, thresholds: List[float] | None = None
) -> List[Dict[str, float]]:
    """
    Evaluate multiple thresholds on P(label=1) and return metrics per threshold.
    """
    if thresholds is None:
        thresholds = [x / 100 for x in range(10, 96, 5)]  # 0.10..0.95

    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        acc = accuracy_score(labels, preds)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": float(acc),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
            }
        )
    return rows

REQUIRED_COLS = {
    "postalCode" : "str",
    "city" : "str",
    "description" : "str",
    "label" : "numeric"
}
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def filter_cols(df : pd.DataFrame, cols : list):
    try :
        out = df[cols].copy()
        return out
    except Exception as e :
        raise ValueError(f"Error filtering DataFrame with {cols}: {e}")
        
def validate_and_cast_trainDs(df : pd.DataFrame, req : dict):
    #verify cols
    diff = set(req.keys()) - set(df.columns)
    if len(diff) != 0 :
        raise ValueError(f"Missing cols : {sorted(diff)}")
    filtered_df = filter_cols(df = df, cols = list(req.keys()))
    string_cols = [k for k,v in req.items() if v == "str" ]
    numeric_cols = [k for k,v in req.items() if v == "numeric"]
    #verify cols dtype
    for k,v in req.items():
        if v == "str" and k in string_cols:
            if k == "postalCode":
                s = filtered_df[k].astype("string")
                x = pd.to_numeric(s, errors="coerce")

                s_norm = np.where(
                    x.isna(),
                    s.str.strip(),
                    np.trunc(x).astype("Int64").astype("string")
                )
                filtered_df[k] = pd.Series(s_norm, index=filtered_df.index, dtype="string")
            else:
                filtered_df[k] = filtered_df[k].astype("string")
        elif v == "numeric" and k in numeric_cols:
            if not pd.api.types.is_numeric_dtype(filtered_df[k]):
                filtered_df[k] = pd.to_numeric(filtered_df[k], errors="coerce")
            filtered_df[k] = np.trunc(filtered_df[k]).astype(np.int64)
    try :
        filtered_df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
        if not filtered_df["label"].isin([0, 1]).all():
            bad = filtered_df.loc[~filtered_df["label"].isin([0, 1]), "label"].unique().tolist()
            raise ValueError(f"Found non-binary labels: {bad}")
    except Exception :
        return filtered_df
    return filtered_df
def build_input_text(postalCode: str, city: str, description: str, desc_max_chars: int) -> str:
    """
    What: Build a single text string that contains all signals.
    Why: The model only sees "one sequence of tokens", so we must include structured fields.

    We keep preprocessing LIGHT to preserve cues:
    - Keep punctuation, accents, line breaks (often meaningful).
    - Truncate description to limit compute.
    """
    postalCode = "" if postalCode is None or pd.isna(postalCode) else str(postalCode)
    city = "" if city is None or pd.isna(city) else str(city)
    description = "" if description is None or pd.isna(description) else str(description)

    # light normalization
    description = description.replace("\r\n", "\n").replace("\r", "\n")
    if desc_max_chars > 0:
        description = description[:desc_max_chars]

    # "field tags" help the model separate columns
    return f"[POSTAL] {postalCode}\n[CITY] {city}\n[DESC]\n{description}"
def prepareDataset(data_path : str, req : dict, desc_max_chars: int = 2000):
    try :
        raw_df = pd.read_csv(data_path)
    except :
        raise ValueError(f"Failed to open ds {data_path} *only csv file*")
    try :
        valid_df = validate_and_cast_trainDs(df = raw_df, req = req)
    except Exception as e :
        raise ValueError(f"Failed to valid and cast dataset. {e}")
    valid_df["input_text"] = valid_df.apply(lambda r :
        build_input_text(postalCode = r.get("postalCode",""),
                        city = r.get("city",""),
                        description = r.get("description",""),
                        desc_max_chars = desc_max_chars
                        ),
                       axis = 1
                      )
    return valid_df
class flagLocation_dataset(torch.utils.data.Dataset):
    """
    encodings example :
    encodings = {
      "input_ids": [
        [10, 11, 12, 0],  # row 0
        [20, 21,  0, 0],  # row 1
      ],
      "attention_mask": [
        [1, 1, 1, 0],     # row 0
        [1, 1, 0, 0],     # row 1
      ]
    }
    each input_id is a sentence that's tokenized and assigned to an integer.
    each attention_mask is a mask to indicate which token is padding vs real. padding is to artficially size the text length to max length.
    """
    def __init__(self, encodings : dict[str, list[list[int]]], labels : np.ndarray):
        #data loading
        self.encodings = encodings
        self.labels = labels.astype(int)
    def __getitem__(self,index):
        #get data sample at index
        item = {k : torch.tensor(v[index]) for k,v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item
    def __len__(self):
        #get dataset length
        return len(self.labels)
def get_tokenizer_and_model(model_name : str, num_labels : int):
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)
    return tokenizer,model

def load_model(model_dir: str):
    """Load a saved tokenizer + sequence classification model from disk."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model
def train_model(
    model,
    tokenizer,
    train_df,
    out_dir,
    val_df=None,
    lr=2e-5,
    train_bs=16,
    eval_bs=32,
    epochs=3,
    weight_decay=0.01,
    seed=42,
    fp16=None,
    logging_steps=50,
    max_len=256,
):
    use_fp16 = torch.cuda.is_available() if fp16 is None else fp16

    # encode train
    train_enc = tokenizer(
        train_df["input_text"].tolist(),
        truncation=True,
        max_length=max_len,
    )
    train_ds = flagLocation_dataset(train_enc, train_df["label"].values)

    # optional val
    val_ds = None
    if val_df is not None and not val_df.empty:
        val_enc = tokenizer(
            val_df["input_text"].tolist(),
            truncation=True,
            max_length=max_len,
        )
        val_ds = flagLocation_dataset(val_enc, val_df["label"].values)

    eval_strategy = "epoch" if val_ds is not None else "no"

    train_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        evaluation_strategy=eval_strategy,
        save_strategy=eval_strategy,
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        fp16=use_fp16,
        report_to="none",
        logging_steps=logging_steps,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if val_ds is not None else None,
    )

    trainer.train()
    metrics = trainer.evaluate() if val_ds is not None else None
    # threshold report on validation if available
    if val_ds is not None:
        pred_out = trainer.predict(val_ds)      
        logits = pred_out.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        labels = val_df["label"].values
        t_rows = threshold_report(probs, labels)
        best = max(t_rows, key=lambda x: x["f1"])
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "threshold_report.json"), "w", encoding="utf-8") as f:
            json.dump({"rows": t_rows, "best_by_f1": best}, f, ensure_ascii=False, indent=2)
    return trainer, metrics
def save_model(trainer,tokenizer,out_dir:str):
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
def get_trainedModel(
    csv_path: str = "ressources/data/location_match_dataset_2000.csv",
    out_dir: str = "",
    model_name: str = "cmarkea/distilcamembert-base",
    max_len: int = 256,
    desc_max_chars: int = 2000,
    test_size: float = 0.15,
    seed: int = 7,
    epochs: int = 3,
    lr: float = 2e-5,
    train_bs: int = 16,
    eval_bs: int = 32,
    weight_decay: float = 0.01,
):
    set_seed(seed)

    # validate/prepare data
    df = prepareDataset(
        data_path=csv_path,
        req=REQUIRED_COLS,
        desc_max_chars=desc_max_chars,
    )
    # tokenizer + model
    tokenizer, model = get_tokenizer_and_model(
        model_name=model_name,
        num_labels=len(df["label"].unique()),
    )
    # split
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    # train via helper
    trainer, metrics = train_model(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=val_df,
        out_dir=out_dir,
        lr=lr,
        train_bs=train_bs,
        eval_bs=eval_bs,
        epochs=epochs,
        weight_decay=weight_decay,
        seed=seed,
    )
    # save
    save_model(trainer=trainer,tokenizer=tokenizer,out_dir=out_dir)
    print("\nSaved model to:", out_dir)
    return trainer, tokenizer, metrics

def load_and_train_model(
    model_dir: str,
    csv_path: str = "ressources/data/location_match_dataset_2000.csv",
    out_dir: str | None = None,
    max_len: int = 256,
    desc_max_chars: int = 2000,
    test_size: float = 0.15,
    seed: int = 7,
    epochs: int = 3,
    lr: float = 2e-5,
    train_bs: int = 16,
    eval_bs: int = 32,
    weight_decay: float = 0.01,
):
    """
    Load an existing saved model/tokenizer from `model_dir`, continue training on `csv_path`,
    and save to `out_dir` (defaults to model_dir if not provided).
    Returns (trainer, metrics, tokenizer, model).
    """
    set_seed(seed)
    if out_dir is None:
        out_dir = model_dir

    df = prepareDataset(data_path=csv_path, req=REQUIRED_COLS, desc_max_chars=desc_max_chars)

    tokenizer, model = load_model(model_dir)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    trainer, metrics = train_model(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=val_df,
        out_dir=out_dir,
        lr=lr,
        train_bs=train_bs,
        eval_bs=eval_bs,
        epochs=epochs,
        weight_decay=weight_decay,
        seed=seed,
        max_len=max_len,
    )
    save_model(trainer=trainer, tokenizer=tokenizer, out_dir=out_dir)
    print("\nSaved model to:", out_dir)
    return trainer, tokenizer, metrics
def predict(input_model, tokenizer, input_text, max_len=256, thresh=0.85):
    """
    Predict using either a Trainer (uses trainer.model) or a raw model.
    Returns (flag, prob) where flag is 1 if prob >= thresh else 0.
    """
    model = getattr(input_model, "model", input_model)
    enc = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_len)
    device = model.device
    enc = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        p_contradiction = probs[0, 1].item()
    return (1 if p_contradiction >= thresh else 0), p_contradiction
