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
    "Age",  # Age is an age-band, not an exact numeric age.
    "CYP2C9 consensus",
    "VKORC1 -1639 consensus",
]

BINARY_FEATURES = [
    "Diabetes",
    "Congestive Heart Failure and/or Cardiomyopathy",
    "Valve Replacement",
    "Current Smoker",
]

ALL_STAGE4_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def validate_feature_groups(approved_features: list[str]) -> None:
    """Fail fast if preprocessing and feature policy disagree."""
    if len(ALL_STAGE4_FEATURES) != len(set(ALL_STAGE4_FEATURES)):
        raise ValueError("A feature appears in more than one preprocessing group.")

    approved = set(approved_features)
    grouped = set(ALL_STAGE4_FEATURES)
    if approved != grouped:
        missing = sorted(approved - grouped)
        extra = sorted(grouped - approved)
        raise ValueError(
            "Preprocessing groups do not match feature_policy.yaml. "
            f"Missing from preprocessing={missing}; extra={extra}"
        )


def build_preprocessor(approved_features: list[str]) -> Pipeline:
    """Build an unfitted preprocessing pipeline.

    Rules:
    - Height/Weight: median imputation.
    - Gender/Age/genotypes: explicit '__MISSING__' category + one-hot encoding.
    - Binary clinical flags: missing becomes -1, then one-hot encoded, so NA != 0.
    - After encoding, standardize all numerical columns to comparable scale.

    The returned object MUST be fit only on a training fold.
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

    # PCA in GĐ5 should see comparable scales across the encoded dimensions.
    return Pipeline(
        steps=[
            ("columns", columns),
            ("scaler", StandardScaler()),
        ]
    )
