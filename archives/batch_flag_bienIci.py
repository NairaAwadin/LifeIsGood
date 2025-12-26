"""
Batch predict contradictions on bienIci_2458.csv using the latest model.

Outputs augmented data with a new column `falseLocation` to `ressources/out_data/`
in both CSV and Parquet formats.
"""
from __future__ import annotations
import time
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import pandas as pd

from get_DistilCamemBERT import (
    prepareDataset,
    load_model,
    predict_with_model_or_trainer,
)

DATA_PATH = "ressources/data/bienIci_2458.csv"
OUT_DIR = Path("ressources/out_data")
OUT_CSV = OUT_DIR / "flagged.csv"
OUT_PARQUET = OUT_DIR / "flagged.parquet"
INFER_COLS = {
    "postalCode": "str",
    "city": "str",
    "description": "str",
}


def find_latest_model_dir(models_root: Path = Path("models")) -> Path:
    """Pick the newest model directory that contains a config.json."""
    candidates = []
    for d in models_root.iterdir():
        if d.is_dir() and (d / "config.json").exists():
            candidates.append((d.stat().st_mtime, d))
    if not candidates:
        raise FileNotFoundError(f"No model with config.json found under {models_root}")
    candidates.sort(reverse=True)  # newest first
    return candidates[0][1]


def batch_predict(
    model,
    tokenizer,
    texts: Iterable[str],
    batch_size: int = 32,
    max_len: int = 256,
    thresh: float = 0.90,
) -> List[int]:
    """Run batched prediction for speed; returns list of flags (0/1)."""
    flags: List[int] = []
    device = model.device
    model.eval()
    with torch.no_grad():
        batch: List[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) < batch_size:
                continue
            enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            flags.extend((probs >= thresh).long().cpu().tolist())
            batch = []
        if batch:
            enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            flags.extend((probs >= thresh).long().cpu().tolist())
    return flags


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_dir = find_latest_model_dir()
    print(f"Using model: {model_dir}")
    tokenizer, model = load_model(str(model_dir))

    df = prepareDataset(data_path=DATA_PATH, req=INFER_COLS, desc_max_chars=2000)

    # batch predict for latency
    flags = batch_predict(
        model=model,
        tokenizer=tokenizer,
        texts=df["input_text"].tolist(),
        batch_size=32,
        max_len=256,
        thresh=0.85,
    )
    df["falseLocation"] = flags

    df.to_csv(OUT_CSV, index=False)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved flagged data to {OUT_CSV} and {OUT_PARQUET}")
    elapsed = time.perf_counter() - t0
    print(f"Elapsed: {elapsed:.3f}s")

if __name__ == "__main__":
    main()
