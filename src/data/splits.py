"""K-Fold split utilities with reproducible saved indices.

Given a DataFrame and split config, produce per-fold train/val index lists,
serialize them under data/splits/{split_name}/seed{S}.json, and provide
loaders for downstream training scripts.

Two split modes:
  - stratified_kfold        : StratifiedKFold on a composite stratify key
  - stratified_group_kfold  : StratifiedGroupKFold on `group_field` (company)

The composite stratify key joins multiple fields with '|' so that rare
joint patterns are still distributed across folds when feasible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def _composite_key(df: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    return df[list(fields)].astype(str).agg("|".join, axis=1).to_numpy()


def make_folds(
    df: pd.DataFrame,
    n_splits: int,
    stratify_fields: Sequence[str],
    seed: int,
    mode: str = "stratified_kfold",
    group_field: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, val_idx) numpy arrays."""
    y = _composite_key(df, stratify_fields)

    if mode == "stratified_kfold":
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(skf.split(np.zeros(len(df)), y))
    elif mode == "stratified_group_kfold":
        if group_field is None or group_field not in df.columns:
            raise ValueError(f"group_field '{group_field}' missing for grouped split")
        groups = df[group_field].astype(str).to_numpy()
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(sgkf.split(np.zeros(len(df)), y, groups=groups))
    else:
        raise ValueError(f"unknown split mode: {mode}")

    return [(np.asarray(tr), np.asarray(va)) for tr, va in folds]


def save_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
    out_dir: str | Path,
    seed: int,
    meta: dict | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "n_splits": len(folds),
        "meta": meta or {},
        "folds": [
            {"fold": i, "train_idx": tr.tolist(), "val_idx": va.tolist()}
            for i, (tr, va) in enumerate(folds)
        ],
    }
    p = out / f"seed{seed}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def load_folds(path: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    return [
        (np.asarray(f["train_idx"]), np.asarray(f["val_idx"]))
        for f in payload["folds"]
    ]


def report_distribution(
    df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], fields: Sequence[str]
) -> dict:
    """Return per-fold per-field label distribution for sanity checks."""
    out: dict = {"n_folds": len(folds), "fields": {}}
    for field in fields:
        out["fields"][field] = []
        for i, (_, va) in enumerate(folds):
            counts = df.iloc[va][field].value_counts().to_dict()
            out["fields"][field].append({"fold": i, "n_val": int(len(va)), "counts": counts})
    return out
