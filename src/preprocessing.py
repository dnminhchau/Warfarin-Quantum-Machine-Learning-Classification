"""Tiền xử lý: mã hóa đặc trưng theo dược di truyền, phục hồi VKORC1 và điền giá trị thiếu bằng hồi quy.

Mọi bước có học tham số nằm trong sklearn Pipeline nên chỉ được fit trên phần huấn luyện.

Thứ tự các bước:
    dữ liệu thô -> PGxFeatureBuilder (6 cột số, phục hồi VKORC1 từ SNP liên kết)
                -> RegressionImputer (điền Height/Weight bằng hồi quy)
                -> chọn k cột (domain / mi / pls / pca) -> điền trung vị
                -> chuẩn hóa (cổ điển) hoặc đưa về góc [0, π] (lượng tử)
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Thứ tự ưu tiên theo thuật toán IWPC 2009, cố định trước khi xem kết quả.
# Biểu diễn domain với k cột lấy k cột đầu của danh sách này.
DOMAIN_ORDER = ["VKORC1", "Weight", "Age", "CYP2C9", "Height", "Male"]

# Điểm hoạt tính CYP2C9 theo CPIC 2017: chức năng bình thường 1, giảm 0.5, mất chức năng 0.
CYP2C9_SCORE = {"*1": 1.0, "*2": 0.5, "*3": 0.0, "*5": 0.5, "*6": 0.0,
                "*8": 0.5, "*11": 0.5, "*13": 0.0, "*14": 0.5}


def vkorc1_allele_count(v) -> float:
    """Số allele A của VKORC1 -1639 (0, 1 hoặc 2); allele A làm tăng nhạy cảm với warfarin."""
    return {"G/G": 0.0, "A/G": 1.0, "G/A": 1.0, "A/A": 2.0}.get(v, np.nan)


def cyp2c9_activity(v) -> float:
    """Tổng điểm hoạt tính của 2 allele CYP2C9, ví dụ *1/*3 = 1.0."""
    if not isinstance(v, str):
        return np.nan
    parts = [p.strip() for p in v.split("/")]
    if len(parts) != 2 or any(p not in CYP2C9_SCORE for p in parts):
        return np.nan
    return CYP2C9_SCORE[parts[0]] + CYP2C9_SCORE[parts[1]]


def age_decade(v) -> float:
    """Đổi nhóm tuổi thành số thập niên: '60 - 69' -> 6, '90+' -> 9 (như công thức IWPC)."""
    if not isinstance(v, str):
        return np.nan
    v = v.strip()
    if v.startswith("90"):
        return 9.0
    return float(int(v.split("-")[0]) // 10) if v[:1].isdigit() else np.nan


class PGxFeatureBuilder(BaseEstimator, TransformerMixin):
    """Tạo 6 cột số theo DOMAIN_ORDER từ dữ liệu thô.

    Khi VKORC1 -1639 bị thiếu, giá trị được suy ra từ SNP liên kết VKORC1 1173 (rs9934438).
    Transformer không học tham số; giá trị thiếu để NaN và được điền ở bước sau.
    use_snp_recovery=False tắt bước phục hồi.
    """

    def __init__(self, use_snp_recovery: bool = True):
        self.use_snp_recovery = use_snp_recovery

    def fit(self, X, y=None):
        return self

    def _vkorc1(self, X: pd.DataFrame) -> np.ndarray:
        base = X["VKORC1 -1639 consensus"].map(vkorc1_allele_count).to_numpy(float)
        if not self.use_snp_recovery or "VKORC1 1173 consensus" not in X.columns:
            return base
        link = X["VKORC1 1173 consensus"]  # 1173 C>T liên kết chặt với -1639
        recover = {"C/C": 0.0, "T/T": 2.0, "C/T": 1.0, "T/C": 1.0}
        out = base.copy()
        for i, (idx, b) in enumerate(zip(X.index, base)):
            if np.isnan(b):
                out[i] = recover.get(link.loc[idx], np.nan)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        num = lambda c: pd.to_numeric(X[c], errors="coerce")
        out = pd.DataFrame(index=X.index)
        out["VKORC1"] = self._vkorc1(X)
        out["Weight"] = num("Weight (kg)")
        out["Age"] = X["Age"].map(age_decade).astype(float)
        out["CYP2C9"] = X["CYP2C9 consensus"].map(cyp2c9_activity)
        out["Height"] = num("Height (cm)")
        out["Male"] = X["Gender"].map({"male": 1.0, "female": 0.0})
        return out[DOMAIN_ORDER]


class RegressionImputer(BaseEstimator, TransformerMixin):
    """Điền Height và Weight bằng hồi quy tuyến tính; giá trị còn thiếu điền trung vị.

    Height ~ Weight + Age + Male; Weight ~ Height + Age + Male.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X, columns=DOMAIN_ORDER)
        self.medians_ = X.median(numeric_only=True)
        self.models_ = {}
        for tgt, src in [("Height", ["Weight", "Age", "Male"]), ("Weight", ["Height", "Age", "Male"])]:
            m = X[tgt].notna()
            if m.sum() >= 20:
                Xs = X.loc[m, src].fillna(self.medians_[src])
                self.models_[tgt] = (src, LinearRegression().fit(Xs, X.loc[m, tgt]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=DOMAIN_ORDER).copy()
        for tgt, (src, lr) in self.models_.items():
            miss = X[tgt].isna()
            if miss.any():
                X.loc[miss, tgt] = lr.predict(X.loc[miss, src].fillna(self.medians_[src]))
        # Giá trị không điền được bằng hồi quy dùng trung vị của tập huấn luyện
        return X.fillna(self.medians_)


class TakeFirstK(BaseEstimator, TransformerMixin):
    """Giữ k cột đầu theo DOMAIN_ORDER."""

    def __init__(self, k: int = 4):
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, : self.k]


class PLSDA(BaseEstimator, TransformerMixin):
    """PLS-DA: giảm xuống k thành phần có tương quan cao nhất với nhãn."""

    def __init__(self, n_components: int = 4):
        self.n_components = n_components

    def fit(self, X, y):
        self.pls_ = PLSRegression(n_components=self.n_components, scale=True).fit(X, np.eye(3)[np.asarray(y, int)])
        return self

    def transform(self, X):
        return self.pls_.transform(X)


def _angle():
    """Đưa mỗi cột về khoảng [0, π] làm góc quay qubit; giá trị ngoài khoảng bị cắt."""
    return MinMaxScaler(feature_range=(0.0, np.pi), clip=True)


def build_representation(name: str = "domain", k: int = 4, use_snp: bool = True, angle: bool = True):
    """Danh sách bước tiền xử lý cho Pipeline (chưa gồm mô hình).

    name: 'domain' (mặc định), 'mi' (mutual information), 'pls' hoặc 'pca', áp dụng trên 6 cột domain.
    angle=True cho mô hình lượng tử, False cho mô hình cổ điển.
    """
    base = [("feats", PGxFeatureBuilder(use_snp)), ("reg_impute", RegressionImputer()),
            ("impute", SimpleImputer(strategy="median"))]
    if name == "domain":
        reduce = [("reduce", TakeFirstK(k))]
    elif name == "mi":
        reduce = [("scale0", StandardScaler()),
                  ("reduce", SelectKBest(partial(mutual_info_classif, random_state=0), k=k))]
    elif name == "pls":
        reduce = [("scale0", StandardScaler()), ("reduce", PLSDA(k))]
    elif name == "pca":
        reduce = [("scale0", StandardScaler()), ("reduce", PCA(k, random_state=0))]
    else:
        raise ValueError(name)
    tail = [("angle", _angle())] if angle else [("scale", StandardScaler())]
    return base + reduce + tail
