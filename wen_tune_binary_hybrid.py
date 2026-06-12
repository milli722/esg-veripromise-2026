from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.dataset import ID2LABEL, LABEL2ID


TRAIN_CSV = Path("data/raw/vpesg4k_train_1000 V1.csv")
TEST_CSV = Path("vpesg4k_test_2000.csv")

BASE_TEST_PROBS = Path(
    "outputs/submissions/wen_combo_v2_test_probs.npz"
)

SPECIALIST_PROBS = Path(
    "outputs/binary_specialist/"
    "wen_binary_specialist_oof_and_test_probs.npz"
)

OUT_DIR = Path("outputs/submissions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBMISSION_COLUMNS = [
    "id",
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]


def argmax_excluding(
    row: np.ndarray,
    excluded_index: int,
) -> int:
    values = row.copy()
    values[excluded_index] = -np.inf
    return int(values.argmax())


def binary_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        f1_score(
            y_true.astype(int),
            y_pred.astype(int),
            zero_division=0,
        )
    )


def tune_thresholds(
    train_df: pd.DataFrame,
    specialist: dict[str, np.ndarray],
) -> tuple[dict, dict, pd.DataFrame]:
    true_t1 = (
        train_df["promise_status"].astype(str) == "Yes"
    ).to_numpy()

    true_t3 = (
        train_df["evidence_status"].astype(str) == "Yes"
    ).to_numpy()

    oof_t1 = specialist["oof_promise_yes"]
    oof_t3 = specialist["oof_evidence_yes"]

    rows = []

    # 不做極端 threshold，降低過度適應 OOF 的風險。
    grid = np.arange(0.30, 0.701, 0.01)

    for t1_threshold in grid:
        pred_t1 = oof_t1 >= t1_threshold

        for t3_threshold in grid:
            # 官方階層邏輯：
            # T1=No 時，T3 必須為 N/A，因此不能預測為 Yes。
            pred_t3 = pred_t1 & (oof_t3 >= t3_threshold)

            t1_f1 = binary_f1(true_t1, pred_t1)
            t3_f1 = binary_f1(true_t3, pred_t3)

            binary_score = (
                0.20 * t1_f1
                + 0.30 * t3_f1
            )

            rows.append({
                "t1_threshold": round(
                    float(t1_threshold), 2
                ),
                "t3_threshold": round(
                    float(t3_threshold), 2
                ),
                "t1_f1": t1_f1,
                "t3_f1": t3_f1,
                "binary_weighted_score": binary_score,
                "oof_t1_yes_rate": pred_t1.mean(),
                "oof_t3_yes_rate": pred_t3.mean(),
            })

    results = (
        pd.DataFrame(rows)
        .sort_values(
            "binary_weighted_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    exact_best = results.iloc[0].to_dict()

    # 避免只挑中一個尖銳的局部最高點：
    # 在距最佳分數 0.001 以內的平坦區域中，
    # 選擇最靠近該區域中位數的設定。
    plateau = results[
        results["binary_weighted_score"]
        >= exact_best["binary_weighted_score"] - 0.001
    ].copy()

    median_t1 = plateau["t1_threshold"].median()
    median_t3 = plateau["t3_threshold"].median()

    plateau["distance_to_plateau_center"] = (
        (plateau["t1_threshold"] - median_t1) ** 2
        + (plateau["t3_threshold"] - median_t3) ** 2
    )

    robust_best = (
        plateau
        .sort_values(
            [
                "distance_to_plateau_center",
                "binary_weighted_score",
            ],
            ascending=[True, False],
        )
        .iloc[0]
        .to_dict()
    )

    return exact_best, robust_best, results


def build_submission(
    name: str,
    t1_threshold: float,
    t3_threshold: float,
    test_df: pd.DataFrame,
    base: dict[str, np.ndarray],
    specialist: dict[str, np.ndarray],
) -> Path:
    test_t1 = specialist["test_promise_yes"]
    test_t3 = specialist["test_evidence_yes"]

    timeline_na = LABEL2ID[
        "verification_timeline"
    ]["N/A"]

    quality_na = LABEL2ID[
        "evidence_quality"
    ]["N/A"]

    rows = []

    for i, row in test_df.iterrows():
        promise = (
            "Yes"
            if test_t1[i] >= t1_threshold
            else "No"
        )

        if promise == "No":
            timeline = "N/A"
            evidence = "N/A"
            quality = "N/A"

        else:
            timeline_idx = argmax_excluding(
                base["verification_timeline"][i],
                timeline_na,
            )

            timeline = ID2LABEL[
                "verification_timeline"
            ][timeline_idx]

            evidence = (
                "Yes"
                if test_t3[i] >= t3_threshold
                else "No"
            )

            if evidence == "No":
                quality = "N/A"

            else:
                quality_idx = argmax_excluding(
                    base["evidence_quality"][i],
                    quality_na,
                )

                quality = ID2LABEL[
                    "evidence_quality"
                ][quality_idx]

        rows.append({
            "id": str(row["id"]),
            "promise_status": promise,
            "verification_timeline": timeline,
            "evidence_status": evidence,
            "evidence_quality": quality,
        })

    submission = pd.DataFrame(rows)[SUBMISSION_COLUMNS]

    # repo 內部名稱與官方提交名稱不同。
    submission["verification_timeline"] = submission[
        "verification_timeline"
    ].replace({
        "longer_than_5_years": "more_than_5_years"
    })

    validate_submission(submission)

    out_path = OUT_DIR / f"{name}.csv"

    submission.to_csv(
        out_path,
        index=False,
        encoding="utf-8",
    )

    print(f"\n[wrote] {out_path}")
    print(
        "T1 predicted Yes-rate:",
        f"{(submission['promise_status'] == 'Yes').mean():.2%}",
    )
    print(
        "T3 predicted Yes-rate:",
        f"{(submission['evidence_status'] == 'Yes').mean():.2%}",
    )

    return out_path


def validate_submission(
    submission: pd.DataFrame,
) -> None:
    allowed_t1 = {"Yes", "No"}

    allowed_t2 = {
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
        "N/A",
    }

    allowed_t3 = {"Yes", "No", "N/A"}

    allowed_t4 = {
        "Clear",
        "Not Clear",
        "Misleading",
        "N/A",
    }

    assert list(submission.columns) == SUBMISSION_COLUMNS
    assert len(submission) == 2000
    assert submission["id"].iloc[0] == "12001"
    assert submission["id"].iloc[-1] == "14000"
    assert not submission.isna().any().any()
    assert not (submission == "").any().any()

    assert set(
        submission["promise_status"]
    ).issubset(allowed_t1)

    assert set(
        submission["verification_timeline"]
    ).issubset(allowed_t2)

    assert set(
        submission["evidence_status"]
    ).issubset(allowed_t3)

    assert set(
        submission["evidence_quality"]
    ).issubset(allowed_t4)

    no_promise = (
        submission["promise_status"] == "No"
    )

    assert (
        submission.loc[
            no_promise,
            "verification_timeline",
        ] == "N/A"
    ).all()

    assert (
        submission.loc[
            no_promise,
            "evidence_status",
        ] == "N/A"
    ).all()

    assert (
        submission.loc[
            no_promise,
            "evidence_quality",
        ] == "N/A"
    ).all()

    no_evidence = (
        submission["evidence_status"] != "Yes"
    )

    assert (
        submission.loc[
            no_evidence,
            "evidence_quality",
        ] == "N/A"
    ).all()


def print_result(
    title: str,
    result: dict,
) -> None:
    print(f"\n========== {title} ==========")

    for key in [
        "t1_threshold",
        "t3_threshold",
        "t1_f1",
        "t3_f1",
        "binary_weighted_score",
        "oof_t1_yes_rate",
        "oof_t3_yes_rate",
    ]:
        print(f"{key}: {result[key]:.6f}")


def main() -> None:
    for path in [
        TRAIN_CSV,
        TEST_CSV,
        BASE_TEST_PROBS,
        SPECIALIST_PROBS,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"缺少檔案：{path}"
            )

    train_df = pd.read_csv(
        TRAIN_CSV,
        dtype=str,
    )

    test_df = pd.read_csv(
        TEST_CSV,
        dtype=str,
    )

    base_npz = np.load(BASE_TEST_PROBS)
    base = {
        key: base_npz[key]
        for key in base_npz.files
    }

    specialist_npz = np.load(SPECIALIST_PROBS)
    specialist = {
        key: specialist_npz[key]
        for key in specialist_npz.files
    }

    exact_best, robust_best, results = (
        tune_thresholds(
            train_df,
            specialist,
        )
    )

    print_result("OOF 精確最高點", exact_best)
    print_result("OOF 平坦區穩健點", robust_best)

    print("\n========== Top 10 ==========")

    print(
        results.head(10).to_string(
            index=False
        )
    )

    results.to_csv(
        OUT_DIR / "wen_binary_threshold_search.csv",
        index=False,
        encoding="utf-8",
    )

    build_submission(
        name="wen_binary_hybrid_exact",
        t1_threshold=exact_best["t1_threshold"],
        t3_threshold=exact_best["t3_threshold"],
        test_df=test_df,
        base=base,
        specialist=specialist,
    )

    build_submission(
        name="wen_binary_hybrid_robust",
        t1_threshold=robust_best["t1_threshold"],
        t3_threshold=robust_best["t3_threshold"],
        test_df=test_df,
        base=base,
        specialist=specialist,
    )


if __name__ == "__main__":
    main()
