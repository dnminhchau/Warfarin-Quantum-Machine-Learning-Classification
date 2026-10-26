
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.make_labels import label_dose

ROOT = Path(__file__).resolve().parent.parent

raw = pd.read_excel(
    ROOT / "data/raw/PS206767-553247439.xls",
    sheet_name="Subject Data",
    engine="xlrd"
)

labels = pd.read_csv(
    ROOT / "data/interim/labels.csv"
)

balance = pd.read_csv(
    ROOT / "docs/results/class_balance.csv"
)

stable = pd.to_numeric(
    raw["Subject Reached Stable Dose of Warfarin"],
    errors="coerce"
)

dose = pd.to_numeric(
    raw["Therapeutic Dose of Warfarin"],
    errors="coerce"
)

# 1. Recalculate eligibility independently
eligible_mask = (
    stable.eq(1)
    & dose.notna()
    & np.isfinite(dose)
    & dose.gt(0)
)

expected_indices = set(raw.index[eligible_mask])

actual_indices = labels["raw_row_index"]

assert actual_indices.notna().all()
assert actual_indices.is_unique

actual_indices = actual_indices.astype(int)

assert len(labels) == eligible_mask.sum()
assert set(actual_indices) == expected_indices

print("[PASS] Eligibility and row indices")

# 2. Verify every label against raw dose
wrong_labels = 0

for _, row in labels.iterrows():
    idx = int(row["raw_row_index"])
    expected_label = label_dose(dose.loc[idx])

    if row["target_group"] != expected_label:
        wrong_labels += 1

assert wrong_labels == 0

print("[PASS] All labels match raw dose")

# 3. Verify class balance
expected_counts = (
    labels["target_group"]
    .value_counts()
    .to_dict()
)

assert set(expected_counts) == {
    "Low", "Intermediate", "High"
}

for _, row in balance.iterrows():
    group = row["target_group"]

    assert group in expected_counts
    assert int(row["count"]) == expected_counts[group]

    expected_pct = (
        expected_counts[group] / len(labels) * 100
    )

    assert abs(
        float(row["percentage"]) - expected_pct
    ) <= 0.01

assert len(balance) == 3
assert balance["count"].sum() == len(labels)

print("[PASS] Class distribution")

print("\n=== GD3 LABEL VERIFICATION ===")
print("Raw records:", len(raw))
print("Eligible records:", eligible_mask.sum())
print("Labeled records:", len(labels))
print("Label mismatches:", wrong_labels)
print("Class counts:", expected_counts)
print("\nALL CHECKS PASSED")
