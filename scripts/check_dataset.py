"""Kiểm tra dữ liệu gốc, nhãn và các tập chia cố định (không huấn luyện mô hình).

Chạy: python -m scripts.check_dataset
Kết quả: results/data_validation/dataset_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_labels import label_dose
from src.dataset import load_dataset, load_policy_columns

ROOT = Path(__file__).resolve().parent.parent


def verify_dataset(root: Path = ROOT) -> dict:
    problems = []
    source = json.loads((root / "data/source_manifest.json").read_text(encoding="utf-8"))
    raw_path = root / source["raw_file"]["path"]
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if raw_hash != source["raw_file"]["sha256"]:
        problems.append("Raw workbook checksum mismatch")

    raw = pd.read_excel(raw_path, sheet_name="Subject Data", engine="xlrd")
    labels = pd.read_csv(root / "data/interim/labels.csv")
    stable = pd.to_numeric(raw["Subject Reached Stable Dose of Warfarin"], errors="coerce")
    dose = pd.to_numeric(raw["Therapeutic Dose of Warfarin"], errors="coerce")
    eligible = stable.eq(1) & dose.gt(0) & np.isfinite(dose)
    if labels.raw_row_index.duplicated().any():
        problems.append("Duplicate label row IDs")
    else:
        expected = dose[eligible].map(label_dose)
        actual = labels.set_index("raw_row_index").target_group
        if set(actual.index) != set(expected.index) or not actual.reindex(expected.index).equals(expected):
            problems.append("Label eligibility or dose classification mismatch")
        for column in ["PharmGKB Subject ID", "PharmGKB Sample ID"]:
            a = raw[column].reindex(labels.raw_row_index).reset_index(drop=True).astype("string").fillna("<NA>")
            b = labels[column].astype("string").fillna("<NA>")
            if not a.equals(b):
                problems.append(f"Patient alignment mismatch: {column}")

    split_dir = root / "data/splits"
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    checksums = {}
    for name, expected_hash in manifest["files"].items():
        content = (split_dir / name).read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        # Manifest được tạo với file CSV xuống dòng kiểu CRLF; chấp nhận cả bản LF có cùng nội dung.
        crlf = content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        status = "exact" if actual_hash == expected_hash else (
            "matches_after_crlf_normalization" if hashlib.sha256(crlf).hexdigest() == expected_hash else "mismatch")
        checksums[name] = {"status": status, "actual_sha256": actual_hash, "manifest_sha256": expected_hash}
        if status == "mismatch":
            problems.append(f"Split checksum mismatch: {name}")

    split = pd.read_csv(split_dir / "final_split.csv")
    dev = set(split.loc[split.partition.eq("development"), "raw_row_index"])
    test = set(split.loc[split.partition.eq("final_test"), "raw_row_index"])
    if split.raw_row_index.duplicated().any() or dev & test or dev | test != set(labels.raw_row_index):
        problems.append("Invalid development/test partition")
    if set(split.partition) != {"development", "final_test"}:
        problems.append("Unknown partition")
    pool = pd.read_csv(split_dir / "quantum_dev_pool.csv")
    pool_ids = set(pool.raw_row_index)
    if pool.raw_row_index.duplicated().any() or not pool_ids <= dev or len(pool_ids) != manifest["quantum_dev_pool_size"]:
        problems.append("Invalid quantum development pool")
    for name, expected_ids in [("development_folds.csv", dev), ("quantum_dev_folds.csv", pool_ids)]:
        folds = pd.read_csv(split_dir / name)
        if (folds.raw_row_index.duplicated().any() or set(folds.raw_row_index) != expected_ids
                or set(folds.fold_id) != set(range(manifest["n_splits"]))):
            problems.append(f"Invalid locked folds: {name}")

    df = load_dataset(root)
    return {"status": "passed" if not problems else "failed", "problems": problems,
            "raw_sha256": raw_hash, "eligible": len(labels), "development": len(dev),
            "final_test": len(test), "quantum_pool": len(pool_ids),
            "class_counts": {str(k): int(v) for k, v in df.y.value_counts().sort_index().items()},
            "complete_cases": int(df[load_policy_columns(root)].notna().all(axis=1).sum()),
            "split_checksums": checksums, "model_training_performed": False}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = verify_dataset()
    out = ROOT / "results/data_validation/dataset_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
