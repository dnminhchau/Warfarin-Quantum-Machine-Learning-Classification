"""Tiền xử lý dựa trên 10 cột được duyệt; dùng khi tạo các tập chia (scripts/prepare_splits.py)."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "Height (cm)",
    "Weight (kg)",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Age",  # nhóm tuổi, không phải tuổi chính xác
    "CYP2C9 consensus",
    "VKORC1 -1639 consensus",
]

BINARY_FEATURES = [
    "Diabetes",
    "Congestive Heart Failure and/or Cardiomyopathy",
    "Valve Replacement",
    "Current Smoker",
]

POLICY_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def validate_feature_groups(approved_features: list[str]) -> None:
    """Báo lỗi nếu các nhóm cột không khớp với configs/feature_policy.yaml."""
    if len(POLICY_FEATURES) != len(set(POLICY_FEATURES)):
        raise ValueError("A feature appears in more than one preprocessing group.")

    approved = set(approved_features)
    grouped = set(POLICY_FEATURES)
    if approved != grouped:
        missing = sorted(approved - grouped)
        extra = sorted(grouped - approved)
        raise ValueError(
            "Preprocessing groups do not match feature_policy.yaml. "
            f"Missing from preprocessing={missing}; extra={extra}"
        )


def build_preprocessor(approved_features: list[str]) -> Pipeline:
    """Pipeline tiền xử lý chưa fit; chỉ fit trên phần huấn luyện.

    - Height, Weight: điền trung vị.
    - Gender, Age, kiểu gen: thêm nhóm '__MISSING__', sau đó one-hot.
    - Cờ lâm sàng nhị phân: giá trị thiếu thành -1 rồi one-hot, để phân biệt thiếu với 0.
    - Sau khi mã hóa, chuẩn hóa toàn bộ các cột.
    """
    validate_feature_groups(approved_features)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    binary_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=-1),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    columns = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
            ("binary", binary_pipe, BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("columns", columns),
            ("scaler", StandardScaler()),
        ]
    )
