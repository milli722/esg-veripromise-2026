from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import (
    ESGDataset,
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    TASKS,
    esg_collate,
)
from src.models.multitask import MultiTaskClassifier


TEST_CSV = "vpesg4k_test_2000.csv"
EXP_NAME = "my_combo_v2_seed42"
SEED_DIR = "seed42"
N_FOLDS = 5

BACKBONE = "hfl/chinese-macbert-base"
POOLING = "cls_mean"
DROPOUT = 0.1
MAX_LENGTH = 384
BATCH_SIZE = 32

OUT_PATH = Path("outputs/submissions/wen_combo_v2_submission.csv")

SUBMISSION_COLUMNS = [
    "id",
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]


def load_test_records() -> list[dict]:
    df = pd.read_csv(TEST_CSV, dtype=str)

    required = {"id", "data"}
    if not required.issubset(df.columns):
        raise ValueError(f"測試檔缺少必要欄位：{required - set(df.columns)}")

    if len(df) != 2000:
        raise ValueError(f"測試資料應為 2000 筆，目前為 {len(df)} 筆")

    return [
        {"id": int(row["id"]), "data": str(row["data"])}
        for row in df.to_dict("records")
    ]


@torch.no_grad()
def infer_one_fold(
    ckpt_path: Path,
    loader: DataLoader,
    device: torch.device,
    n_records: int,
) -> dict[str, np.ndarray]:
    model = MultiTaskClassifier(
        backbone=BACKBONE,
        num_labels=NUM_LABELS,
        pooling=POOLING,
        dropout=DROPOUT,
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    fold_probs = {
        task: np.zeros((n_records, NUM_LABELS[task]), dtype=np.float64)
        for task in TASKS
    }

    with torch.amp.autocast(
        device_type=device.type,
        enabled=(device.type == "cuda"),
    ):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            indices = batch["_index"].cpu().numpy()

            logits = model(input_ids, attention_mask)

            for task in TASKS:
                probs = torch.softmax(logits[task].float(), dim=-1)
                fold_probs[task][indices] = probs.cpu().numpy()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return fold_probs


def argmax_without(probs: np.ndarray, excluded_index: int) -> int:
    masked = probs.copy()
    masked[excluded_index] = -np.inf
    return int(masked.argmax())


def decode_predictions(
    avg_probs: dict[str, np.ndarray],
    records: list[dict],
) -> list[dict]:
    timeline_na = LABEL2ID["verification_timeline"]["N/A"]
    evidence_na = LABEL2ID["evidence_status"]["N/A"]
    quality_na = LABEL2ID["evidence_quality"]["N/A"]
    evidence_yes = LABEL2ID["evidence_status"]["Yes"]

    results = []

    for i, record in enumerate(records):
        promise_idx = int(avg_probs["promise_status"][i].argmax())
        promise = ID2LABEL["promise_status"][promise_idx]

        row = {
            "id": record["id"],
            "promise_status": promise,
        }

        # 官方邏輯：沒有承諾，其他欄位全部為 N/A
        if promise == "No":
            row["verification_timeline"] = "N/A"
            row["evidence_status"] = "N/A"
            row["evidence_quality"] = "N/A"

        else:
            timeline_idx = argmax_without(
                avg_probs["verification_timeline"][i],
                timeline_na,
            )
            row["verification_timeline"] = ID2LABEL[
                "verification_timeline"
            ][timeline_idx]

            evidence_idx = argmax_without(
                avg_probs["evidence_status"][i],
                evidence_na,
            )
            row["evidence_status"] = ID2LABEL[
                "evidence_status"
            ][evidence_idx]

            # 官方邏輯：沒有佐證，品質必須為 N/A
            if evidence_idx == evidence_yes:
                quality_idx = argmax_without(
                    avg_probs["evidence_quality"][i],
                    quality_na,
                )
                row["evidence_quality"] = ID2LABEL[
                    "evidence_quality"
                ][quality_idx]
            else:
                row["evidence_quality"] = "N/A"

        results.append(row)

    return results


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    records = load_test_records()
    print(f"[data] loaded {len(records)} test rows")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

    dataset = ESGDataset(
        records,
        tokenizer,
        max_length=MAX_LENGTH,
        with_labels=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=esg_collate,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    avg_probs = {
        task: np.zeros((len(records), NUM_LABELS[task]), dtype=np.float64)
        for task in TASKS
    }

    for fold in range(N_FOLDS):
        ckpt_path = (
            Path("outputs/checkpoints")
            / EXP_NAME
            / SEED_DIR
            / f"fold{fold}"
            / "best.pt"
        )

        if not ckpt_path.exists():
            raise FileNotFoundError(f"缺少 checkpoint：{ckpt_path}")

        print(f"[infer] fold {fold + 1}/{N_FOLDS}")

        fold_probs = infer_one_fold(
            ckpt_path,
            loader,
            device,
            len(records),
        )

        for task in TASKS:
            avg_probs[task] += fold_probs[task]

    for task in TASKS:
        avg_probs[task] /= N_FOLDS

    # 儲存平均後的測試機率，之後可快速產生 threshold 候選，
    # 不必重新做 5-fold 推論。
    probs_path = Path("outputs/submissions/wen_combo_v2_test_probs.npz")
    probs_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(probs_path, **avg_probs)
    print(f"[wrote] {probs_path}")

    rows = decode_predictions(avg_probs, records)
    submission = pd.DataFrame(rows)[SUBMISSION_COLUMNS]

    # repo 內部使用 longer_than_5_years；
    # 官方提交格式要求 more_than_5_years。
    submission["verification_timeline"] = submission[
        "verification_timeline"
    ].replace({"longer_than_5_years": "more_than_5_years"})

    submission = submission.sort_values("id").reset_index(drop=True)

    # 結構檢查
    assert len(submission) == 2000
    assert list(submission.columns) == SUBMISSION_COLUMNS
    assert submission["id"].iloc[0] == 12001
    assert submission["id"].iloc[-1] == 14000
    assert not submission.isna().any().any()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"\n[wrote] {OUT_PATH}")
    print(f"[rows] {len(submission)}")

    print("\n[label distribution]")
    for col in SUBMISSION_COLUMNS[1:]:
        print(f"\n{col}")
        print(submission[col].value_counts().to_string())


if __name__ == "__main__":
    main()
