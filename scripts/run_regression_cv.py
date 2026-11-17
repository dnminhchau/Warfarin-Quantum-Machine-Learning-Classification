"""Hồi quy liều warfarin trên Development, đánh giá bằng MAE, PW20 và R².

Mô hình dự đoán căn bậc hai của liều; liều dự đoán được đổi thành 3 nhóm để tính thêm Macro-F1.
Chạy: python -m scripts.run_regression_cv --k 4
Kết quả: results/regression_cv/regression_k<k>.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.baselines import iwpc_predict_dose
from src.dataset import load_dataset, site_labels, split_ids
from src.evaluation import regression_metrics, to_class
from src.preprocessing import build_representation
from src.quantum_models import feature_states, make_feature_map
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=4); a = ap.parse_args()
    df = load_dataset(ROOT)
    dev = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]
    y = dev["y"].to_numpy(); dose = dev["dose"].to_numpy()
    skf = StratifiedKFold(5, shuffle=True, random_state=42); st = site_labels(dev)
    agg = {m: {"mae": [], "pw20": [], "r2": [], "f1": []} for m in ["Ridge", "SVR", "QSVR(quantum)", "IWPC-formula"]}
    fm = make_feature_map(a.k)
    for tr, va in skf.split(dev, st):
        clf = Pipeline(build_representation("domain", a.k, True, angle=False)).fit(dev.iloc[tr])
        Xtr, Xva = clf.transform(dev.iloc[tr]), clf.transform(dev.iloc[va])
        ang = Pipeline(build_representation("domain", a.k, True, angle=True)).fit(dev.iloc[tr])
        Atr, Ava = ang.transform(dev.iloc[tr]), ang.transform(dev.iloc[va])
        ytr_s = np.sqrt(dose[tr]); true = dose[va]
        preds = {"Ridge": Ridge(alpha=1.0).fit(Xtr, ytr_s).predict(Xva) ** 2,
                 "SVR": SVR(C=10, epsilon=1.0).fit(Xtr, ytr_s).predict(Xva) ** 2}
        S, Sv = feature_states(0.25 * Atr, fm), feature_states(0.25 * Ava, fm)
        preds["QSVR(quantum)"] = SVR(kernel="precomputed", C=10, epsilon=1.0).fit(
            np.abs(S.conj() @ S.T) ** 2, ytr_s).predict(np.abs(Sv.conj() @ S.T) ** 2) ** 2
        preds["IWPC-formula"] = iwpc_predict_dose(dev.iloc[va])
        for m, p in preds.items():
            rm = regression_metrics(true, p)
            agg[m]["mae"].append(rm["mae_mg_week"]); agg[m]["pw20"].append(rm["pw20"]); agg[m]["r2"].append(rm["r2"])
            agg[m]["f1"].append(f1_score(y[va], to_class(p), average="macro"))
    rows = [{"model": m, "MAE": np.mean(v["mae"]), "PW20": np.mean(v["pw20"]),
             "R2": np.mean(v["r2"]), "MacroF1_3class": np.mean(v["f1"])} for m, v in agg.items()]
    out = ROOT / "results" / "regression_cv"; out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows).sort_values("MAE"); res.to_csv(out / f"regression_k{a.k}.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
