"""Tạo nhãn nhóm liều từ dữ liệu IWPC gốc.

Giữ bệnh nhân đã đạt liều ổn định và có liều hợp lệ (> 0), rồi gán nhãn:
Low (<= 21 mg/tuần), Intermediate (21-49), High (>= 49).
Chạy: python -m scripts.prepare_labels
Kết quả: data/interim/labels.csv và results/data_validation/class_balance.csv
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    ROOT / "data" / "raw"
    / "PS206767-553247439.xls"
)

INTERIM_DIR = ROOT / "data" / "interim"
RESULTS_DIR = ROOT / "results" / "data_validation"

STABLE_COL = "Subject Reached Stable Dose of Warfarin"
DOSE_COL = "Therapeutic Dose of Warfarin"

def label_dose(dose):
    """Gán nhóm liều cho một liều warfarin hợp lệ (mg/tuần)."""

    dose = float(dose)

    if not math.isfinite(dose) or dose <= 0:
        raise ValueError("Dose must be finite and > 0")

    if dose <= 21:
        return "Low"

    if dose < 49:
        return "Intermediate"

    return "High"


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()

    if (INTERIM_DIR / "labels.csv").exists():
        raise FileExistsError("Labels already exist; use scripts.check_dataset to validate them.")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(
        RAW_FILE,
        sheet_name="Subject Data",
        engine="xlrd"
    )

    print("Label construction")
    print("Raw records:", len(df))

    # Chuyển sang dạng số để kiểm tra
    stable = pd.to_numeric(
        df[STABLE_COL],
        errors="coerce"
    )

    dose = pd.to_numeric(
        df[DOSE_COL],
        errors="coerce"
    )

    # Tiêu chí chọn: đạt liều ổn định và liều hợp lệ
    valid_dose = (
        dose.notna()
        & np.isfinite(dose)
        & dose.gt(0)
    )

    eligible_mask = stable.eq(1) & valid_dose

    eligible = df.loc[eligible_mask].copy()

    # Giữ chỉ số dòng gốc để đối chiếu với dữ liệu thô
    eligible["raw_row_index"] = eligible.index

    print("Eligible records:", len(eligible))

    # Gán nhãn
    eligible["target_group"] = (
        dose.loc[eligible_mask].apply(label_dose)
    )

    # Kiểm tra nhãn
    assert eligible["target_group"].notna().all()

    assert set(eligible["target_group"].unique()).issubset(
        {"Low", "Intermediate", "High"}
    )

    assert len(eligible) == int(eligible_mask.sum())

    labels = eligible[
        [
            "raw_row_index",
            "PharmGKB Subject ID",
            "PharmGKB Sample ID",
            "target_group"
        ]
    ].copy()

    labels.to_csv(
        INTERIM_DIR / "labels.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Phân bố các nhóm
    class_order = [
        "Low",
        "Intermediate",
        "High"
    ]

    counts = (
        eligible["target_group"]
        .value_counts()
        .reindex(class_order, fill_value=0)
    )

    class_balance = pd.DataFrame({
        "target_group": class_order,
        "count": counts.values
    })

    class_balance["percentage"] = (
        class_balance["count"]
        / len(eligible)
        * 100
    ).round(2)

    class_balance.to_csv(
        RESULTS_DIR / "class_balance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nClass distribution")
    print(class_balance.to_string(index=False))

    print("\nTotal labeled:", counts.sum())
    print("Labels saved to data/interim/labels.csv")
    print("Class balance saved to results/data_validation/class_balance.csv")


if __name__ == "__main__":
    main()
