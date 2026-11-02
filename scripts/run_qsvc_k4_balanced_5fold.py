from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.algorithms import QSVC

from scripts.prepare_stage4 import (
    load_and_validate_dataset,
    verify_raw_file,
)
from src.preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/results"
SPLITS = ROOT / "data/splits"

OUTPUTS = [
    "qsvc_k4_balanced_5fold_metrics.csv",
    "qsvc_k4_balanced_5fold_predictions.csv",
    "qsvc_k4_balanced_5fold_confusion.csv",
    "qsvc_k4_balanced_5fold_config.json",
]

CLASSES = ["Low", "Intermediate", "High"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    existing = [
        name for name in OUTPUTS
        if (OUT / name).exists()
    ]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite: {existing}"
        )

    with (
        ROOT / "configs/feature_policy.yaml"
    ).open(encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    approved = policy["approved_X_columns"]
    assert policy["training_allowed"] is True
    assert len(approved) == 10

    verify_raw_file()

    manifest_path = SPLITS / "split_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["approved_features"] == approved
    assert manifest["status"] == (
        "LOCKED_BEFORE_MODEL_PERFORMANCE"
    )

    for filename, expected in manifest["files"].items():
        actual = sha256(SPLITS / filename)
        assert actual == expected, (
            f"Split checksum mismatch: {filename}"
        )

    X, y = load_and_validate_dataset(approved)

    labels = pd.read_csv(
        ROOT / "data/interim/labels.csv"
    )
    folds = pd.read_csv(
        SPLITS / "quantum_dev_folds.csv"
    )
    partitions = pd.read_csv(
        SPLITS / "final_split.csv"
    )

    assert len(X) == len(y) == len(labels) == 5410
    assert np.array_equal(
        y.to_numpy(),
        labels["target_group"].to_numpy()
    )
    assert labels["raw_row_index"].is_unique
    assert folds["raw_row_index"].is_unique
    assert set(folds["fold_id"]) == set(range(5))

    pool = set(folds["raw_row_index"])
    dev = set(partitions.loc[
        partitions.partition == "development",
        "raw_row_index"
    ])
    test = set(partitions.loc[
        partitions.partition == "final_test",
        "raw_row_index"
    ])

    assert len(pool) == 600
    assert pool.issubset(dev)
    assert pool.isdisjoint(test)
    assert dev.isdisjoint(test)
    assert dev | test == set(labels["raw_row_index"])

    metrics = []
    predictions = []
    confusion_rows = []
    total_cm = np.zeros((3, 3), dtype=int)

    print("[PASS] Raw and split checksums")
    print("[PASS] Final Test excluded")

    for fold_id in range(5):
        val_ids = set(folds.loc[
            folds.fold_id == fold_id,
            "raw_row_index"
        ])
        train_ids = pool - val_ids

        assert len(train_ids) == 480
        assert len(val_ids) == 120

        train_pos = np.flatnonzero(
            labels["raw_row_index"]
            .isin(train_ids).to_numpy()
        )
        val_pos = np.flatnonzero(
            labels["raw_row_index"]
            .isin(val_ids).to_numpy()
        )

        X_train = X.iloc[train_pos]
        X_val = X.iloc[val_pos]
        y_train = y.iloc[train_pos]
        y_val = y.iloc[val_pos]

        prep = build_preprocessor(approved)
        train_encoded = prep.fit_transform(X_train)
        val_encoded = prep.transform(X_val)

        pca = PCA(
            n_components=4,
            svd_solver="full"
        )
        train_pca = pca.fit_transform(train_encoded)
        val_pca = pca.transform(val_encoded)

        scaler = MinMaxScaler(
            feature_range=(0.0, float(np.pi))
        )
        z_train = np.clip(
            scaler.fit_transform(train_pca),
            0.0, np.pi
        )
        z_val = np.clip(
            scaler.transform(val_pca),
            0.0, np.pi
        )

        assert z_train.shape == (480, 4)
        assert z_val.shape == (120, 4)
        assert np.isfinite(z_train).all()
        assert np.isfinite(z_val).all()

        feature_map = zz_feature_map(
            feature_dimension=4,
            reps=2,
            entanglement="linear"
        )

        kernel = FidelityStatevectorKernel(
            feature_map=feature_map,
            shots=None
        )

        model = QSVC(
            quantum_kernel=kernel,
            C=1.0,
            class_weight="balanced"
        )

        start = perf_counter()
        model.fit(z_train, y_train)
        fit_time = perf_counter() - start

        start = perf_counter()
        pred = model.predict(z_val)
        predict_time = perf_counter() - start

        recalls = recall_score(
            y_val, pred,
            labels=CLASSES,
            average=None,
            zero_division=0
        )

        cm = confusion_matrix(
            y_val, pred,
            labels=CLASSES
        )
        total_cm += cm

        row = {
            "fold": fold_id,
            "k": 4,
            "train_n": 480,
            "val_n": 120,
            "macro_f1": f1_score(
                y_val, pred,
                labels=CLASSES,
                average="macro",
                zero_division=0
            ),
            "accuracy": accuracy_score(y_val, pred),
            "balanced_accuracy":
                balanced_accuracy_score(y_val, pred),
            "recall_low": float(recalls[0]),
            "recall_mid": float(recalls[1]),
            "recall_high": float(recalls[2]),
            "pred_high": int((pred == "High").sum()),
            "support_vectors":
                int(model.n_support_.sum()),
            "pca_explained_variance":
                float(pca.explained_variance_ratio_.sum()),
            "fit_seconds": fit_time,
            "predict_seconds": predict_time,
        }

        metrics.append(row)

        for position, actual, predicted in zip(
            val_pos, y_val, pred
        ):
            predictions.append({
                "raw_row_index": int(
                    labels.iloc[position]["raw_row_index"]
                ),
                "fold": fold_id,
                "actual": actual,
                "predicted": predicted,
            })

        for i, actual in enumerate(CLASSES):
            for j, predicted in enumerate(CLASSES):
                confusion_rows.append({
                    "scope": "fold",
                    "fold": fold_id,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(cm[i, j]),
                })

        print(
            f"[PASS] Fold {fold_id}: "
            f"Macro-F1={row['macro_f1']:.4f}, "
            f"fit={fit_time:.3f}s"
        )

    for i, actual in enumerate(CLASSES):
        for j, predicted in enumerate(CLASSES):
            confusion_rows.append({
                "scope": "aggregate",
                "fold": "all",
                "actual": actual,
                "predicted": predicted,
                "count": int(total_cm[i, j]),
            })

    assert len(predictions) == 600
    assert len({
        row["raw_row_index"] for row in predictions
    }) == 600

    config = {
        "experiment": "E02_QSVC_k4_balanced",
        "dataset": "quantum_dev_pool_600",
        "folds": 5,
        "pca_k": 4,
        "feature_map": "zz_feature_map",
        "reps": 2,
        "entanglement": "linear",
        "kernel": "FidelityStatevectorKernel",
        "shots": None,
        "C": 1.0,
        "class_weight": "balanced",
        "features": approved,
        "preprocessing": "fit_on_training_fold",
        "quantum_scaling": "MinMax_0_pi_then_clip",
        "final_test_used": False,
        "versions": {
            name: version(name)
            for name in [
                "qiskit",
                "qiskit-machine-learning",
                "scikit-learn",
                "numpy",
                "pandas",
            ]
        },
        "checksums": {
            "split_manifest": sha256(manifest_path),
            "feature_policy": sha256(
                ROOT / "configs/feature_policy.yaml"
            ),
            "script": sha256(Path(__file__)),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metrics).to_csv(
        OUT / OUTPUTS[0], index=False
    )
    pd.DataFrame(predictions).to_csv(
        OUT / OUTPUTS[1], index=False
    )
    pd.DataFrame(confusion_rows).to_csv(
        OUT / OUTPUTS[2], index=False
    )

    with (OUT / OUTPUTS[3]).open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            config, f,
            indent=2,
            ensure_ascii=False
        )

    report = pd.DataFrame(metrics)

    print("\n=== QSVC K=4 - 5-FOLD SUMMARY ===")
    print(
        report[
            ["macro_f1", "accuracy",
             "recall_high", "fit_seconds"]
        ].agg(["mean", "std"]).round(4)
    )

    print("\n=== AGGREGATED CONFUSION MATRIX ===")
    print(pd.DataFrame(
        total_cm,
        index=CLASSES,
        columns=CLASSES
    ))

    print("\n[PASS] QSVC results saved to docs/results/")


if __name__ == "__main__":
    main()
