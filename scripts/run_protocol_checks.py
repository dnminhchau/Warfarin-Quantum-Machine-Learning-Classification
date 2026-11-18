"""Các phân tích phụ trong docs/protocol.md, chỉ trên tập Development.

--part outliers:  CV phân tầng theo trung tâm sau khi bỏ bệnh nhân có liều >= 100 mg/tuần, so với toàn bộ.
--part iwpc_hold: công thức IWPC dùng 10 cột được duyệt so với bản thêm chủng tộc và thuốc dùng kèm (cột HOLD).
Chạy: python -m scripts.run_protocol_checks --part outliers
Kết quả: results/subgroup_analysis/outlier_sensitivity.csv, results/regression_cv/iwpc_hold_columns.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import iwpc_predict_dose
from src.dataset import load_dataset, split_ids
from src.evaluation import classif_zoo, classification_metrics, cv_site_stratified, regression_metrics, to_class

ROOT = Path(__file__).resolve().parent.parent
OUTLIER_DOSE = 100


def development() -> pd.DataFrame:
    df = load_dataset(ROOT)
    return df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]


def part_outliers(a):
    dev = development()
    cohorts = {"all": dev, "no_outliers": dev[~(dev["dose"] >= OUTLIER_DOSE)]}
    print(f"Bỏ {len(dev) - len(cohorts['no_outliers'])} bệnh nhân có liều >= {OUTLIER_DOSE} mg/tuần")
    zoo = classif_zoo(); rows = []
    for name, sub in cohorts.items():
        for m in a.models:
            model, grid, kind = zoo[m]
            for r in cv_site_stratified(sub, model, grid, kind, "domain", 4, n_jobs=a.n_jobs):
                rows.append({"cohort": name, "model": m, "n": len(sub), **r, "best_params": str(r["best_params"])})
            cur = pd.DataFrame(rows).query("cohort == @name and model == @m")
            print(f"{name:12s} {m:13s} MacroF1={cur.macro_f1.mean():.3f}±{cur.macro_f1.std():.3f}", flush=True)
    out = ROOT / "results" / "subgroup_analysis"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "outlier_sensitivity.csv", index=False)


def part_iwpc_hold(a):
    dev = development()
    y, dose = dev["y"].to_numpy(), dev["dose"].to_numpy(float)
    rows = []
    for hold in (False, True):
        pred = iwpc_predict_dose(dev, use_hold_columns=hold)
        rows.append({"variant": "10 cột + chủng tộc, thuốc dùng kèm" if hold else "10 cột được duyệt",
                     "n": len(dev), **classification_metrics(y, to_class(pred)), **regression_metrics(dose, pred)})
    res = pd.DataFrame(rows)
    out = ROOT / "results" / "regression_cv"; out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "iwpc_hold_columns.csv", index=False)
    print(res.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=["outliers", "iwpc_hold"], required=True)
    ap.add_argument("--models", nargs="*", default=["LR", "QSVC-product"])
    ap.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args()
    {"outliers": part_outliers, "iwpc_hold": part_iwpc_hold}[a.part](a)


if __name__ == "__main__":
    main()
