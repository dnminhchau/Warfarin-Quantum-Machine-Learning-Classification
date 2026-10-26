
from pathlib import Path
import math
import numpy as np
import pandas as pd

# =====================================
# 1. CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    ROOT / "data" / "raw"
    / "PS206767-553247439.xls"
)

INTERIM_DIR = ROOT / "data" / "interim"
DOCS_DIR = ROOT / "docs" / "results"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

STABLE_COL = "Subject Reached Stable Dose of Warfarin"
DOSE_COL = "Therapeutic Dose of Warfarin"

# =====================================
# 2. LABEL FUNCTION
# =====================================

def label_dose(dose):
    """Classify a valid weekly Warfarin dose."""

    dose = float(dose)

    if not math.isfinite(dose) or dose <= 0:
        raise ValueError("Dose must be finite and > 0")

    if dose <= 21:
        return "Low"

    if dose < 49:
        return "Intermediate"

    return "High"


# =====================================
# 3. MAIN PIPELINE
# =====================================

def main():

    df = pd.read_excel(
        RAW_FILE,
        sheet_name="Subject Data",
        engine="xlrd"
    )

    print("=== LABEL CONSTRUCTION ===")
    print("Raw records:", len(df))

    # Convert values for validation
    stable = pd.to_numeric(
        df[STABLE_COL],
        errors="coerce"
    )

    dose = pd.to_numeric(
        df[DOSE_COL],
        errors="coerce"
    )

    # Same eligibility policy as GĐ3 Step 1
    valid_dose = (
        dose.notna()
        & np.isfinite(dose)
        & dose.gt(0)
    )

    eligible_mask = stable.eq(1) & valid_dose

    eligible = df.loc[eligible_mask].copy()

    # Preserve original DataFrame row index
    eligible["raw_row_index"] = eligible.index

    print("Eligible records:", len(eligible))

    # Generate labels
    eligible["target_group"] = (
        dose.loc[eligible_mask].apply(label_dose)
    )

    # Validate generated labels
    assert eligible["target_group"].notna().all()

    assert set(eligible["target_group"].unique()).issubset(
        {"Low", "Intermediate", "High"}
    )

    assert len(eligible) == int(eligible_mask.sum())

    # =====================================
    # 4. EXPORT LABELS
    # =====================================

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

    # =====================================
    # 5. CLASS DISTRIBUTION
    # =====================================

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
        DOCS_DIR / "class_balance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== CLASS DISTRIBUTION ===")
    print(class_balance.to_string(index=False))

    print("\nTotal labeled:", counts.sum())
    print("Labels saved to data/interim/labels.csv")
    print("Class balance saved to docs/results/class_balance.csv")


if __name__ == "__main__":
    main()
