from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BACKBONE = "hfl/chinese-macbert-base"
TRAIN_CSV = "data/raw/vpesg4k_train_1000 V1.csv"
OUT_DIR = Path("outputs/tapt/wen_macbert_esg_train_only")


class MLMTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=384):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_special_tokens_mask=True,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        return {
            key: value[index]
            for key, value in self.encodings.items()
        }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV, dtype=str)
    texts = (
        df["data"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    texts = [
        text
        for text in texts.drop_duplicates().tolist()
        if text
    ]

    print(f"[corpus] {len(texts)} unique ESG paragraphs")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModelForMaskedLM.from_pretrained(BACKBONE)

    dataset = MLMTextDataset(
        texts,
        tokenizer,
        max_length=384,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "_trainer"),
        num_train_epochs=8,
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("[tapt] starting train-only MLM continued pretraining")
    trainer.train()

    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    print(f"[wrote] {OUT_DIR}")


if __name__ == "__main__":
    main()
