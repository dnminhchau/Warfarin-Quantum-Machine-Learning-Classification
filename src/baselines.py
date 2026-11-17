"""Mô hình cổ điển và công thức IWPC dùng làm mốc so sánh."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from src.quantum_models import QuantumKernelSVC, VQCClassifier


class BalancedMLP(BaseEstimator, ClassifierMixin):
    """MLPClassifier có trọng số mẫu cân bằng theo lớp (MLP không hỗ trợ class_weight)."""

    def __init__(self, hidden_layer_sizes=(32, 16), alpha=1e-4, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                  max_iter=1000, early_stopping=True, n_iter_no_change=20,
                                  random_state=self.random_state)
        self.mlp_.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        self.classes_ = self.mlp_.classes_
        return self

    def predict(self, X):
        return self.mlp_.predict(X)


# Danh sách mô hình và lưới tham số của bộ benchmark trong results/benchmarks.
# Mỗi mô hình tối đa 15 cấu hình, chọn bằng inner CV theo Macro-F1.
def model_zoo(vqc_seeds=(0,)) -> dict:
    zoo = {
        "Dummy": (DummyClassifier(strategy="most_frequent"), {}),
        "LR": (LogisticRegression(max_iter=5000, class_weight="balanced"),
               {"C": [0.01, 0.1, 1, 10, 100]}),
        "SVM": (SVC(kernel="rbf", class_weight="balanced"),
                {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, "scale", 1.0, 10.0]}),
        "MLP": (BalancedMLP(), {"hidden_layer_sizes": [(16,), (32, 16)], "alpha": [1e-4, 1e-2]}),
        "HGB": (HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, random_state=0),
                {"max_depth": [2, 3], "learning_rate": [0.05, 0.1]}),
        "QSVC": (QuantumKernelSVC(feature_map="zz", reps=2),
                 {"bandwidth": [0.05, 0.1, 0.25, 0.5, 1.0], "C": [0.1, 1, 10]}),
        # Đối chứng không rối lượng tử
        "QSVC-product": (QuantumKernelSVC(feature_map="product", reps=2),
                         {"bandwidth": [0.05, 0.1, 0.25, 0.5, 1.0], "C": [0.1, 1, 10]}),
    }
    for s in vqc_seeds:
        zoo[f"VQC_s{s}"] = (VQCClassifier(seed=s, maxiter=300),
                            {"layers": [1, 2], "reps": [1, 3], "bandwidth": [0.5, 1.0]})
    return zoo


QUANTUM_MODELS_PREFIX = ("QSVC", "VQC")


# Công thức dược di truyền IWPC (NEJM 2009) với hệ số đã công bố, không huấn luyện lại.
# Công thức được xây dựng trên chính bộ dữ liệu IWPC nên kết quả có thể lạc quan.
# Mặc định chỉ dùng các biến trong 10 cột được duyệt; chủng tộc, amiodarone và thuốc cảm ứng enzyme = 0.
def iwpc_predict_class(df: pd.DataFrame, use_hold_columns: bool = False) -> np.ndarray:
    """Nhóm liều (0/1/2) theo công thức IWPC."""
    dose = iwpc_predict_dose(df, use_hold_columns)
    return np.where(dose <= 21, 0, np.where(dose >= 49, 2, 1))


def iwpc_predict_dose(df: pd.DataFrame, use_hold_columns: bool = False) -> np.ndarray:
    """Liều dự đoán (mg/tuần) theo công thức IWPC."""
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    age = df["Age"].map(lambda v: float(str(v).strip()[0]) if isinstance(v, str) else np.nan)
    h, w = num("Height (cm)"), num("Weight (kg)")
    # Giá trị thiếu điền bằng trung vị của chính tập đang tính
    age, h, w = age.fillna(age.median()), h.fillna(h.median()), w.fillna(w.median())
    vk, cy = df["VKORC1 -1639 consensus"], df["CYP2C9 consensus"]
    known_cyp = ["*1/*1", "*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3"]
    s = (5.6044 - 0.2614 * age + 0.0087 * h + 0.0128 * w
         - 0.8677 * (vk == "A/G") - 1.6974 * (vk == "A/A") - 0.4854 * vk.isna()
         - 0.5211 * (cy == "*1/*2") - 0.9357 * (cy == "*1/*3") - 1.0616 * (cy == "*2/*2")
         - 1.9206 * (cy == "*2/*3") - 2.3312 * (cy == "*3/*3") - 0.2188 * (~cy.isin(known_cyp)))
    if use_hold_columns:  # chỉ dùng cho phân tích độ nhạy
        race = df["Race (OMB)"].astype(str)
        flag = lambda c: (pd.to_numeric(df[c], errors="coerce") == 1)
        inducer = flag("Carbamazepine (Tegretol)") | flag("Phenytoin (Dilantin)") | flag("Rifampin or Rifampicin")
        s = (s - 0.1092 * race.str.contains("Asian") - 0.2760 * race.str.contains("Black")
             - 0.1032 * race.str.contains("Unknown") + 1.1816 * inducer - 0.5503 * flag("Amiodarone (Cordarone)"))
    return s.to_numpy(float) ** 2  # công thức dự đoán căn bậc hai của liều/tuần
