"""Danh sách mô hình, pipeline, chỉ số đánh giá, cross-validation và đánh giá Final Test.

Mọi mô hình được tinh chỉnh bằng cùng một inner CV theo Macro-F1, với class_weight='balanced'
khi mô hình hỗ trợ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, r2_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from src.baselines import iwpc_predict_class, iwpc_predict_dose
from src.preprocessing import build_representation
from src.quantum_models import (QuantumKernelSVC, VQCClassifier, feature_states,
                                   make_feature_map)

# Mô hình phân loại
def classif_zoo(vqc_seed: int = 0) -> dict:
    """Trả về {tên: (estimator, lưới tham số, loại)}; loại 'quantum' dùng đầu vào dạng góc [0, π]."""
    return {
        "Dummy": (DummyClassifier(strategy="most_frequent"), {}, "classical"),
        "LR": (LogisticRegression(max_iter=5000, class_weight="balanced"),
               {"model__C": [0.01, 0.1, 1, 10, 100]}, "classical"),
        "SVM": (SVC(class_weight="balanced"),
                {"model__C": [0.1, 1, 10], "model__gamma": [0.01, 0.1, "scale", 1.0]}, "classical"),
        "MLP": (MLPClassifier(max_iter=1000, early_stopping=True, random_state=0),
                {"model__hidden_layer_sizes": [(16,), (32, 16)], "model__alpha": [1e-4, 1e-2]}, "classical"),
        "HGB": (HistGradientBoostingClassifier(class_weight="balanced", random_state=0),
                {"model__max_depth": [2, 3], "model__learning_rate": [0.05, 0.1]}, "classical"),
        "QSVC": (QuantumKernelSVC(feature_map="zz"),
                 {"model__bandwidth": [0.05, 0.1, 0.25, 0.5, 1.0], "model__C": [0.1, 1, 10]}, "quantum"),
        "QSVC-product": (QuantumKernelSVC(feature_map="product"),
                         {"model__bandwidth": [0.05, 0.1, 0.25, 0.5, 1.0], "model__C": [0.1, 1, 10]}, "quantum"),
        "VQC": (VQCClassifier(seed=vqc_seed, maxiter=250),
                {"model__layers": [1, 2], "model__reps": [1, 3], "model__bandwidth": [0.5, 1.0]}, "quantum"),
    }


def make_pipeline(model, kind: str, rep: str = "domain", k: int = 4, use_snp: bool = True) -> Pipeline:
    """Pipeline tiền xử lý + mô hình. Mô hình lượng tử nhận góc [0, π], mô hình cổ điển nhận dữ liệu chuẩn hóa."""
    angle = (kind == "quantum")
    return Pipeline(build_representation(rep, k, use_snp, angle) + [("model", clone(model))])


# Chỉ số đánh giá
def classification_metrics(y_true, y_pred) -> dict:
    """Macro-F1, accuracy, recall từng nhóm, QWK và tỉ lệ nhầm Thấp <-> Cao."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rec = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"macro_f1": f1_score(y_true, y_pred, average="macro"),
            "accuracy": accuracy_score(y_true, y_pred),
            "recall_low": rec[0], "recall_intermediate": rec[1], "recall_high": rec[2],
            "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
            "severe_error_low_high": float(np.mean(np.abs(y_true - y_pred) == 2))}


def regression_metrics(dose_true, dose_pred) -> dict:
    """MAE (mg/tuần), PW20 và R² cho dự đoán liều."""
    e = np.abs(dose_pred - dose_true)
    return {"mae_mg_week": float(e.mean()), "pw20": float(np.mean(e <= 0.2 * dose_true)),
            "r2": float(r2_score(dose_true, dose_pred))}


def to_class(dose) -> np.ndarray:
    """Đổi liều (mg/tuần) thành nhóm: <= 21 Thấp, >= 49 Cao, còn lại Vừa."""
    return np.where(dose <= 21, 0, np.where(dose >= 49, 2, 1))


# Cross-validation phân tầng theo trung tâm
def cv_site_stratified(df, model, grid, kind, rep="domain", k=4, n_splits=5, seed=42, n_jobs=2):
    """5-fold phân tầng theo trung tâm trên Development, tinh chỉnh bằng inner 3-fold; trả về chỉ số từng fold."""
    from src.dataset import site_labels
    y = df["y"].to_numpy(); st = site_labels(df)
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for k_, (tr, va) in enumerate(outer.split(df, st)):
        pipe = make_pipeline(model, kind, rep, k)
        if grid:
            gs = GridSearchCV(pipe, grid, scoring="f1_macro",
                              cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=n_jobs)
            gs.fit(df.iloc[tr], y[tr])
            best, bp = gs.best_estimator_, gs.best_params_
        else:
            best, bp = pipe.fit(df.iloc[tr], y[tr]), {}
        rows.append({"fold": k_, **classification_metrics(y[va], best.predict(df.iloc[va])),
                     "best_params": bp})
    return rows


# Final Test
def tuned_predict(df_dev, df_test, model, grid, kind, rep="domain", k=4, seed=42, n_jobs=2):
    """Tinh chỉnh bằng inner CV trên df_dev, huấn luyện lại trên toàn bộ df_dev, dự đoán df_test."""
    y = df_dev["y"].to_numpy()
    pipe = make_pipeline(model, kind, rep, k)
    if grid:
        gs = GridSearchCV(pipe, grid, scoring="f1_macro",
                          cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=n_jobs).fit(df_dev, y)
        return gs.predict(df_test), gs.best_params_
    return pipe.fit(df_dev, y).predict(df_test), {}


def paired_bootstrap(y_true, pred_a, pred_b, n=2000, seed=42):
    """Chênh lệch Macro-F1 (A - B) và khoảng tin cậy 95% bằng paired bootstrap."""
    rng = np.random.default_rng(seed); y = np.asarray(y_true); a = np.asarray(pred_a); b = np.asarray(pred_b)
    d = np.array([f1_score(y[i], a[i], average="macro") - f1_score(y[i], b[i], average="macro")
                  for i in (rng.integers(0, len(y), len(y)) for _ in range(n))])
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "significant": not (lo < 0 < hi)}
