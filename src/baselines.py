"""MLP có cân bằng lớp và công thức IWPC dùng làm mốc so sánh."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight


class BalancedMLP(BaseEstimator, ClassifierMixin):
    """MLPClassifier có trọng số mẫu cân bằng theo lớp.

    MLPClassifier không có tham số class_weight, nên trọng số được truyền qua sample_weight với cùng công thức
    class_weight='balanced' của các mô hình khác. Các thiết lập còn lại giống MLP không cân bằng trong
    src/evaluation.py (max_iter 1000, early stopping, random_state 0).
    """

    def __init__(self, hidden_layer_sizes=(32, 16), alpha=1e-4, random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                  max_iter=1000, early_stopping=True, random_state=self.random_state)
        self.mlp_.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        self.classes_ = self.mlp_.classes_
        return self

    def predict(self, X):
        return self.mlp_.predict(X)


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
