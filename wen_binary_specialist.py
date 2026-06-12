from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


TRAIN_CSV = "data/raw/vpesg4k_train_1000 V1.csv"
TEST_CSV = "vpesg4k_test_2000.csv"

BACKBONE = "hfl/chinese-macbert-base"
OUT_DIR = Path("outputs/binary_specialist")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
SEED = 42
MAX_LENGTH = 384
BATCH_SIZE = 8
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, with_labels: bool):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.with_labels = with_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoded = self.tokenizer(
            str(row["data"]),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        item = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "index": idx,
        }

        if self.with_labels:
            item["t1"] = 1 if row["promise_status"] == "Yes" else 0

            # T3 只在 promise=Yes 的資料中學習 Yes / No。
            # promise=No 時 evidence_status=N/A，不計入 T3 loss。
            if row["promise_status"] == "Yes":
                item["t3"] = 1 if row["evidence_status"] == "Yes" else 0
                item["t3_mask"] = 1.0
            else:
                item["t3"] = 0
                item["t3_mask"] = 0.0

        return item


def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(values, pad_value):
        return [
            v + [pad_value] * (max_len - len(v))
            for v in values
        ]

    out = {
        "input_ids": torch.tensor(
            pad([x["input_ids"] for x in batch], 0),
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            pad([x["attention_mask"] for x in batch], 0),
            dtype=torch.long,
        ),
        "index": torch.tensor([x["index"] for x in batch]),
    }

    if "t1" in batch[0]:
        out["t1"] = torch.tensor(
            [x["t1"] for x in batch],
            dtype=torch.float32,
        )
        out["t3"] = torch.tensor(
            [x["t3"] for x in batch],
            dtype=torch.float32,
        )
        out["t3_mask"] = torch.tensor(
            [x["t3_mask"] for x in batch],
            dtype=torch.float32,
        )

    return out


class BinarySpecialist(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BACKBONE)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.15)
        self.t1_head = nn.Linear(hidden, 1)
        self.t3_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls = out.last_hidden_state[:, 0]
        x = self.dropout(cls)

        return {
            "t1": self.t1_head(x).squeeze(-1),
            "t3": self.t3_head(x).squeeze(-1),
        }


def train_fold(model, loader):
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # T1、T3 正負類別不平衡，使用溫和 pos_weight。
    t1_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([0.8], device=DEVICE)
    )

    t3_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([0.8], device=DEVICE),
        reduction="none",
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    for epoch in range(EPOCHS):
        running = 0.0

        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            t1 = batch["t1"].to(DEVICE)
            t3 = batch["t3"].to(DEVICE)
            t3_mask = batch["t3_mask"].to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda"),
            ):
                logits = model(ids, mask)

                loss_t1 = t1_loss_fn(logits["t1"], t1)

                raw_t3 = t3_loss_fn(logits["t3"], t3)
                loss_t3 = (
                    (raw_t3 * t3_mask).sum()
                    / t3_mask.sum().clamp_min(1.0)
                )

                loss = 0.45 * loss_t1 + 0.55 * loss_t3

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()

        print(
            f"    epoch {epoch + 1}/{EPOCHS} "
            f"loss={running / len(loader):.4f}"
        )


@torch.no_grad()
def infer(model, loader):
    model.eval()

    t1_probs = np.zeros(len(loader.dataset), dtype=np.float64)
    t3_probs = np.zeros(len(loader.dataset), dtype=np.float64)

    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        indices = batch["index"].numpy()

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(DEVICE.type == "cuda"),
        ):
            logits = model(ids, mask)

        t1_probs[indices] = torch.sigmoid(
            logits["t1"]
        ).cpu().numpy()

        t3_probs[indices] = torch.sigmoid(
            logits["t3"]
        ).cpu().numpy()

    return t1_probs, t3_probs


def main():
    print(f"[device] {DEVICE}")

    train_df = pd.read_csv(TRAIN_CSV, dtype=str)
    test_df = pd.read_csv(TEST_CSV, dtype=str)

    print(f"[train] {len(train_df)}")
    print(f"[test]  {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

    stratify = (
        train_df["promise_status"].astype(str)
        + "|"
        + train_df["evidence_status"].astype(str)
    )

    groups = train_df["company"].fillna("unknown").astype(str)

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    test_dataset = TextDataset(
        test_df,
        tokenizer,
        with_labels=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate,
    )

    test_t1 = np.zeros(len(test_df), dtype=np.float64)
    test_t3 = np.zeros(len(test_df), dtype=np.float64)

    for fold, (train_idx, _) in enumerate(
        splitter.split(train_df, stratify, groups),
        start=1,
    ):
        print(f"\n[fold {fold}/{N_SPLITS}]")

        set_seed(SEED + fold)

        fold_df = train_df.iloc[train_idx].reset_index(drop=True)

        train_dataset = TextDataset(
            fold_df,
            tokenizer,
            with_labels=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate,
        )

        model = BinarySpecialist().to(DEVICE)
        train_fold(model, train_loader)

        fold_t1, fold_t3 = infer(model, test_loader)

        test_t1 += fold_t1 / N_SPLITS
        test_t3 += fold_t3 / N_SPLITS

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    out_path = OUT_DIR / "wen_binary_specialist_test_probs.npz"

    np.savez_compressed(
        out_path,
        promise_yes=test_t1,
        evidence_yes=test_t3,
    )

    print(f"\n[wrote] {out_path}")
    print(f"T1 mean P(Yes): {test_t1.mean():.4f}")
    print(f"T3 mean P(Yes): {test_t3.mean():.4f}")


if __name__ == "__main__":
    main()
