"""Đọc dữ liệu IWPC, nhãn và các tập chia đã cố định trong data/splits.

Module không tạo tập chia mới. load_dataset trả về các cột thô cần cho tiền xử lý,
gồm cả các SNP liên kết dùng để phục hồi VKORC1 -1639.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CLASS_ORDER = ["Low", "Intermediate", "High"]  # 0, 1, 2
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}
RAW_NAME = "PS206767-553247439.xls"

# SNP VKORC1 liên kết, chỉ dùng để phục hồi VKORC1 -1639 khi thiếu (không phải đặc trưng).
LINKAGE_SNP = ["VKORC1 1173 consensus", "VKORC1 2255 consensus", "VKORC1 1542 consensus"]
# Cột chỉ dùng cho phân tích, không đưa vào X: trung tâm, chủng tộc, thuốc dùng kèm, liều thực tế.
ANALYSIS_COLS = ["Project Site", "Race (OMB)", "Amiodarone (Cordarone)", "Carbamazepine (Tegretol)",
                 "Phenytoin (Dilantin)", "Rifampin or Rifampicin", "Target INR",
                 "Estimated Target INR Range Based on Indication", "Therapeutic Dose of Warfarin"]


def load_policy_columns(root: Path) -> list[str]:
    """10 cột đầu vào được duyệt trong configs/feature_policy.yaml."""
    policy = yaml.safe_load((root / "configs/feature_policy.yaml").read_text(encoding="utf-8"))
    return list(policy["approved_X_columns"])


def load_dataset(root: Path) -> pd.DataFrame:
    """Bảng 5.410 bệnh nhân hợp lệ, index = raw_row_index, kèm nhãn y và liều dose."""
    raw = pd.read_excel(root / "data/raw" / RAW_NAME, sheet_name="Subject Data", engine="xlrd")
    labels = pd.read_csv(root / "data/interim/labels.csv", encoding="utf-8-sig")
    x_cols = load_policy_columns(root)
    keep = list(dict.fromkeys(x_cols + LINKAGE_SNP + ANALYSIS_COLS))
    df = raw.loc[labels.raw_row_index.to_numpy(), keep].copy()
    df.index = labels.raw_row_index.to_numpy()
    df.index.name = "raw_row_index"
    df["y"] = labels.target_group.map(CLASS_TO_INT).to_numpy()
    df["dose"] = pd.to_numeric(df["Therapeutic Dose of Warfarin"], errors="coerce")
    return df


def split_ids(root: Path, partition: str) -> np.ndarray:
    """raw_row_index của tập 'development' hoặc 'final_test'."""
    fs = pd.read_csv(root / "data/splits/final_split.csv")
    return fs.loc[fs.partition == partition, "raw_row_index"].to_numpy()


def load_folds(root: Path, pool: str) -> pd.DataFrame:
    """Fold cố định cho pool 'qdev600' hoặc 'dev4328'; báo lỗi nếu có bệnh nhân của Final Test."""
    name = {"qdev600": "quantum_dev_folds.csv", "dev4328": "development_folds.csv"}[pool]
    folds = pd.read_csv(root / "data/splits" / name)
    test_ids = set(split_ids(root, "final_test"))
    assert not (set(folds.raw_row_index) & test_ids), "Fold chứa bệnh nhân của Final Test"
    return folds


def site_labels(df: pd.DataFrame) -> np.ndarray:
    """Mã trung tâm (Project Site), dùng để phân tầng khi chia fold."""
    return df["Project Site"].astype(str).to_numpy()


def apply_cohort(df: pd.DataFrame, cohort: str = "all") -> pd.DataFrame:
    """Lọc nhóm bệnh nhân theo đặc điểm đầu vào.

    all: toàn bộ; genotyped: có đủ VKORC1 và CYP2C9; inr23: INR mục tiêu 2-3.
    """
    if cohort == "all":
        return df
    if cohort == "genotyped":
        keep = df["VKORC1 -1639 consensus"].notna() & df["CYP2C9 consensus"].notna()
        return df[keep]
    if cohort == "inr23":
        tn = pd.to_numeric(df["Target INR"], errors="coerce")
        est = df["Estimated Target INR Range Based on Indication"].astype(str).str.strip()
        return df[tn.between(2, 3) | est.eq("2-3")]
    raise ValueError(cohort)


def fold_iter(folds: pd.DataFrame):
    """Sinh (fold_id, id_train, id_val) cho từng fold cố định."""
    ids = folds.raw_row_index.to_numpy()
    for k in sorted(folds.fold_id.unique()):
        yield int(k), ids[folds.fold_id.to_numpy() != k], ids[folds.fold_id.to_numpy() == k]
