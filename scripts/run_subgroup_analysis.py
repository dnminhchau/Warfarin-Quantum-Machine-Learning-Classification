"""Phân tích theo nhóm, phân tích độ nhạy và chọn đặc trưng.

--part subgroup:    CV riêng cho nhóm Da trắng và Châu Á, chỉ gồm bệnh nhân đủ 10 cột.
--part sensitivity: so sánh các nhóm all, genotyped (đủ VKORC1 và CYP2C9), inr23 (INR mục tiêu 2-3).
--part selection:   RFE, forward selection, mutual information và permutation importance.
Chủng tộc chỉ dùng để chia nhóm khi đánh giá, không đưa vào đặc trưng.
Chạy: python -m scripts.run_subgroup_analysis --part subgroup --n-jobs -1
Kết quả: results/subgroup_analysis/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import apply_cohort, load_dataset, load_policy_columns, split_ids
from src.evaluation import classif_zoo, cv_site_stratified
from src.preprocessing import DOMAIN_ORDER, PGxFeatureBuilder

ROOT = Path(__file__).resolve().parent.parent


def dev_df():
    df = load_dataset(ROOT)
    return df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]


def part_subgroup(a):
    dev = dev_df(); x = load_policy_columns(ROOT); cc = dev[x].notna().all(axis=1)
    zoo = classif_zoo()
    rows = []
    for race, sub in [("Da trắng", dev[(dev["Race (OMB)"] == "White") & cc]),
                      ("Châu Á", dev[(dev["Race (OMB)"] == "Asian") & cc])]:
        print(f"\n{race} (đủ 10 cột) n={len(sub)} (Thấp/Vừa/Cao={np.bincount(sub.y, minlength=3).tolist()})")
        for n in ["LR", "SVM", "HGB", "QSVC-product"]:
            model, grid, kind = zoo[n]
            # Nhóm nhỏ nên phân tầng theo nhãn thay vì theo trung tâm
            from sklearn.model_selection import StratifiedKFold, GridSearchCV
            from src.evaluation import make_pipeline, classification_metrics
            y = sub.y.to_numpy(); skf = StratifiedKFold(5, shuffle=True, random_state=42); f1 = []
            for fold, (tr, va) in enumerate(skf.split(sub, y)):
                pipe = make_pipeline(model, kind, "domain", 4)
                gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=3, n_jobs=a.n_jobs).fit(sub.iloc[tr], y[tr]) if grid else pipe.fit(sub.iloc[tr], y[tr])
                metrics = classification_metrics(y[va], gs.predict(sub.iloc[va]))
                f1.append(metrics["macro_f1"])
                rows.append({"subgroup": race, "model": n, "fold": fold, "n": len(sub), **metrics})
            print(f"  {n:13s} MacroF1={np.mean(f1):.3f}±{np.std(f1, ddof=1):.3f}")
    save_rows(rows, "subgroup_metrics.csv")


def part_sensitivity(a):
    dev = dev_df(); zoo = classif_zoo()
    rows = []
    for coh in ["all", "genotyped", "inr23"]:
        sub = apply_cohort(dev, coh)
        print(f"\ncohort={coh} n={len(sub)}")
        for n in ["LR", "QSVC-product"]:
            model, grid, kind = zoo[n]
            r = cv_site_stratified(sub, model, grid, kind, "domain", 4, n_jobs=a.n_jobs)
            f1 = [x["macro_f1"] for x in r]
            rows.extend({"cohort": coh, "model": n, "n": len(sub), **row} for row in r)
            print(f"  {n:13s} MacroF1={np.mean(f1):.3f}±{np.std(f1, ddof=1):.3f}")
    save_rows(rows, "sensitivity_metrics.csv")


def save_rows(rows, filename):
    out = ROOT / "results/subgroup_analysis"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / filename, index=False)


def part_selection(a):
    from src.dataset import load_folds, fold_iter
    dev = dev_df(); folds = load_folds(ROOT, "dev4328")
    enc = PGxFeatureBuilder(use_snp_recovery=True)
    rfe = np.zeros(6); fwd = np.zeros(6); mi = []; perm = []
    for k_, tr, va in fold_iter(folds):
        pre = Pipeline([("i", SimpleImputer(strategy="mean")), ("s", StandardScaler())]).fit(enc.transform(dev.loc[tr]))
        Z = pre.transform(enc.transform(dev.loc[tr])); y = dev.loc[tr, "y"].to_numpy()
        lr = LogisticRegression(max_iter=5000, class_weight="balanced")
        rfe += RFE(lr, n_features_to_select=1).fit(Z, y).ranking_
        fwd += SequentialFeatureSelector(lr, n_features_to_select=4, direction="forward", cv=3,
                                         scoring="f1_macro").fit(Z, y).get_support().astype(int)
        mi.append(mutual_info_classif(Z, y, random_state=0))
        clf = Pipeline([("i", SimpleImputer(strategy="mean")), ("s", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=5000, class_weight="balanced"))]).fit(enc.transform(dev.loc[tr]), y)
        perm.append(permutation_importance(clf, enc.transform(dev.loc[va]), dev.loc[va, "y"].to_numpy(),
                                           scoring="f1_macro", n_repeats=10, random_state=0).importances_mean)
    res = pd.DataFrame({"feature": DOMAIN_ORDER, "RFE_rank(1=best)": (rfe / 5).round(1),
                        "forward_/5": fwd.astype(int), "mutual_info": np.mean(mi, 0).round(4),
                        "perm_importance": np.mean(perm, 0).round(4)}).sort_values("RFE_rank(1=best)")
    out = ROOT / "results" / "subgroup_analysis"; out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "feature_selection.csv", index=False)
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["subgroup", "sensitivity", "selection"], required=True)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    {"subgroup": part_subgroup, "sensitivity": part_sensitivity, "selection": part_selection}[a.part](a)


if __name__ == "__main__":
    main()
