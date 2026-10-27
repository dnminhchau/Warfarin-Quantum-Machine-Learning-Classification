from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.preprocessing import build_preprocessor, validate_feature_groups

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "PS206767-553247439.xls"
LABELS_FILE = ROOT / "data" / "interim" / "labels.csv"
FEATURE_POLICY_FILE = ROOT / "configs" / "feature_policy.yaml"
STAGE4_CONFIG_FILE = ROOT / "configs" / "stage4.yaml"
SOURCE_MANIFEST_FILE = ROOT / "docs" / "reference" / "source_manifest.json"
SPLIT_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "docs" / "results"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_raw_file() -> dict:
    manifest = load_json(SOURCE_MANIFEST_FILE)
    raw_info = manifest["raw_file"]

    if RAW_FILE.name != raw_info["filename"]:
        raise RuntimeError(
            f"Raw filename mismatch: {RAW_FILE.name} != {raw_info['filename']}"
        )
    if not RAW_FILE.exists():
        raise FileNotFoundError(RAW_FILE)

    actual_hash = sha256_file(RAW_FILE)
    expected_hash = raw_info["sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Raw workbook SHA256 mismatch. Stop before creating GĐ4 splits.\n"
            f"Expected: {expected_hash}\nActual:   {actual_hash}"
        )
    return manifest


def load_and_validate_dataset(approved_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_excel(RAW_FILE, sheet_name="Subject Data", engine="xlrd")
    labels = pd.read_csv(LABELS_FILE)

    required_label_cols = {
        "raw_row_index",
        "PharmGKB Subject ID",
        "PharmGKB Sample ID",
        "target_group",
    }
    missing_label_cols = required_label_cols - set(labels.columns)
    if missing_label_cols:
        raise RuntimeError(f"labels.csv missing columns: {sorted(missing_label_cols)}")

    if labels["raw_row_index"].duplicated().any():
        raise RuntimeError("labels.csv contains duplicated raw_row_index values")

    allowed_labels = {"Low", "Intermediate", "High"}
    if not set(labels["target_group"].unique()).issubset(allowed_labels):
        raise RuntimeError("Unexpected target label found")

    row_index = labels["raw_row_index"].astype(int).to_numpy()
    if (row_index < 0).any() or (row_index >= len(raw)).any():
        raise RuntimeError("labels.csv contains raw_row_index outside raw workbook")

    aligned = raw.iloc[row_index].copy().reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    # Strong identity check: GĐ3 labels must still point to the same raw rows.
    for id_col in ["PharmGKB Subject ID", "PharmGKB Sample ID"]:
        left = aligned[id_col].astype("string").fillna("<NA>")
        right = labels[id_col].astype("string").fillna("<NA>")
        if not left.equals(right):
            raise RuntimeError(f"Row alignment check failed for {id_col}")

    missing_features = [c for c in approved_features if c not in raw.columns]
    if missing_features:
        raise RuntimeError(f"Approved feature(s) missing from workbook: {missing_features}")

    X = aligned[approved_features].copy()
    y = labels["target_group"].copy()

    if len(X) != 5410 or len(y) != 5410:
        raise RuntimeError(f"Expected 5410 eligible rows, found X={len(X)}, y={len(y)}")

    return X, y


def stratified_partition_indices(y: pd.Series, config: dict):
    all_pos = np.arange(len(y))
    dev_pos, test_pos = train_test_split(
        all_pos,
        test_size=float(config["final_test_fraction"]),
        random_state=int(config["split_seed"]),
        stratify=y,
        shuffle=True,
    )

    qdev_size = int(config["quantum_dev_pool_size"])
    if qdev_size >= len(dev_pos):
        raise ValueError("quantum_dev_pool_size must be smaller than Development Set")

    qdev_pos, _ = train_test_split(
        dev_pos,
        train_size=qdev_size,
        random_state=int(config["qdev_seed"]),
        stratify=y.iloc[dev_pos],
        shuffle=True,
    )

    return np.sort(dev_pos), np.sort(test_pos), np.sort(qdev_pos)


def build_fold_table(
    positions: np.ndarray,
    labels: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    pool = labels.iloc[positions].copy().reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_id = np.full(len(pool), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(pool, pool["target_group"])):
        fold_id[val_idx] = fold

    if (fold_id < 0).any():
        raise RuntimeError("Some samples did not receive a fold ID")

    return pd.DataFrame(
        {
            "raw_row_index": pool["raw_row_index"].astype(int),
            "fold_id": fold_id,
        }
    ).sort_values("raw_row_index").reset_index(drop=True)


def class_counts(y: pd.Series, positions: np.ndarray, class_order: list[str]) -> dict:
    return (
        y.iloc[positions]
        .value_counts()
        .reindex(class_order, fill_value=0)
        .astype(int)
        .to_dict()
    )


def main():
    verify_raw_file()

    feature_policy = load_yaml(FEATURE_POLICY_FILE)
    config = load_yaml(STAGE4_CONFIG_FILE)

    if not feature_policy.get("training_allowed", False):
        raise RuntimeError("feature_policy.yaml does not allow research training")

    approved_features = list(feature_policy["approved_X_columns"])
    if len(approved_features) != 10:
        raise RuntimeError(f"Expected exactly 10 approved features, found {len(approved_features)}")

    # This also checks that code cannot silently drift from feature_policy.yaml.
    validate_feature_groups(approved_features)

    X, y = load_and_validate_dataset(approved_features)
    labels = pd.read_csv(LABELS_FILE).reset_index(drop=True)

    dev_pos, test_pos, qdev_pos = stratified_partition_indices(y, config)

    # Final split file contains only stable row keys, not patient/sample IDs.
    split = pd.DataFrame(
        {
            "raw_row_index": labels["raw_row_index"].astype(int),
            "partition": "development",
        }
    )
    split.loc[test_pos, "partition"] = "final_test"
    split = split.sort_values("raw_row_index").reset_index(drop=True)
    final_split_path = SPLIT_DIR / "final_split.csv"
    split.to_csv(final_split_path, index=False, encoding="utf-8")

    qdev = pd.DataFrame(
        {"raw_row_index": labels.iloc[qdev_pos]["raw_row_index"].astype(int)}
    ).sort_values("raw_row_index").reset_index(drop=True)
    qdev_path = SPLIT_DIR / "quantum_dev_pool.csv"
    qdev.to_csv(qdev_path, index=False, encoding="utf-8")

    qdev_folds = build_fold_table(
        qdev_pos,
        labels,
        n_splits=int(config["n_splits"]),
        seed=int(config["cv_seed"]),
    )
    qdev_folds_path = SPLIT_DIR / "quantum_dev_folds.csv"
    qdev_folds.to_csv(qdev_folds_path, index=False, encoding="utf-8")

    dev_folds = build_fold_table(
        dev_pos,
        labels,
        n_splits=int(config["n_splits"]),
        seed=int(config["cv_seed"]),
    )
    dev_folds_path = SPLIT_DIR / "development_folds.csv"
    dev_folds.to_csv(dev_folds_path, index=False, encoding="utf-8")

    # Build once only to validate configuration. DO NOT fit here.
    _ = build_preprocessor(approved_features)

    class_order = list(config["class_order"])
    summary_rows = []
    for name, positions in [
        ("all_eligible", np.arange(len(y))),
        ("development", dev_pos),
        ("final_test", test_pos),
        ("quantum_dev_pool", qdev_pos),
    ]:
        counts = class_counts(y, positions, class_order)
        summary_rows.append(
            {
                "partition": name,
                "n": len(positions),
                **{f"n_{c}": counts[c] for c in class_order},
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "stage4_split_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    # Lock the split artifacts with hashes and the exact seeds/configuration.
    split_manifest = {
        "stage": "GĐ4",
        "status": "LOCKED_BEFORE_MODEL_PERFORMANCE",
        "final_test_policy": "Do not use for feature selection, model selection, or hyperparameter tuning.",
        "approved_feature_count": len(approved_features),
        "approved_features": approved_features,
        "final_test_fraction": float(config["final_test_fraction"]),
        "quantum_dev_pool_size": int(config["quantum_dev_pool_size"]),
        "n_splits": int(config["n_splits"]),
        "seeds": {
            "split_seed": int(config["split_seed"]),
            "qdev_seed": int(config["qdev_seed"]),
            "cv_seed": int(config["cv_seed"]),
        },
        "files": {
            "final_split.csv": sha256_file(final_split_path),
            "quantum_dev_pool.csv": sha256_file(qdev_path),
            "quantum_dev_folds.csv": sha256_file(qdev_folds_path),
            "development_folds.csv": sha256_file(dev_folds_path),
        },
    }
    manifest_path = SPLIT_DIR / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)

    # Sanity checks after writing.
    dev_rows = set(split.loc[split["partition"] == "development", "raw_row_index"])
    test_rows = set(split.loc[split["partition"] == "final_test", "raw_row_index"])
    qdev_rows = set(qdev["raw_row_index"])

    assert dev_rows.isdisjoint(test_rows)
    assert len(dev_rows | test_rows) == len(labels)
    assert qdev_rows.issubset(dev_rows)
    assert len(qdev_rows) == int(config["quantum_dev_pool_size"])

    print("=== GĐ4 COMPLETE ===")
    print(summary.to_string(index=False))
    print("\nLocked split artifacts:")
    for p in [final_split_path, qdev_path, qdev_folds_path, dev_folds_path, manifest_path]:
        print(" -", p.relative_to(ROOT))
    print("\nIMPORTANT: preprocessing.py defines an UNFITTED pipeline only.")
    print("Fit a fresh preprocessor inside each training fold in GĐ5/model training.")


if __name__ == "__main__":
    main()
