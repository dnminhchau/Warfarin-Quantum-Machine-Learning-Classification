"""Phân loại nhóm liều trên Development: 5-fold phân tầng theo trung tâm, tinh chỉnh bằng inner 3-fold.

Không dùng Final Test. Mô hình đã có kết quả sẽ được bỏ qua.
Chạy:
    python -m scripts.run_classification_cv --rep domain --k 4 --n-jobs -1
    python -m scripts.run_classification_cv --rep pca --k 4
Kết quả: results/classification_cv/<rep>_k<k>_summary.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import apply_cohort, load_dataset, split_ids
from src.evaluation import classif_zoo, cv_site_stratified

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="domain", choices=["domain", "pca", "pls", "mi"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--cohort", default="all", choices=["all", "genotyped", "inr23"])
    ap.add_argument("--vqc-seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()

    df = load_dataset(ROOT)
    dev = apply_cohort(df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))], a.cohort)
    zoo = classif_zoo(a.vqc_seed)
    names = a.models or list(zoo)
    out = ROOT / "results" / "classification_cv"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.rep}_k{a.k}" + ("" if a.cohort == "all" else f"_{a.cohort}")
    summary_path = out / f"{tag}_summary.csv"
    config_path = out / f"{tag}_config.json"
    rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    completed = {row["model"] for row in rows}
    config = {"rep": a.rep, "k": a.k, "cohort": a.cohort, "n_dev": int(len(dev)),
              "cv": "5-fold stratified by site + inner 3-fold tuning", "final_test_used": False}
    if rows:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in config.items()):
            raise ValueError("Existing result configuration differs; do not merge unrelated experiments.")
        if "VQC" in completed and previous.get("vqc_seed", 0) != a.vqc_seed:
            raise ValueError("Existing VQC uses a different seed.")
    config["vqc_seed"] = a.vqc_seed
    for n in names:
        if n in completed:
            print(f"{n}: existing result; skipped")
            continue
        model, grid, kind = zoo[n]
        r = cv_site_stratified(dev, model, grid, kind, a.rep, a.k, n_jobs=a.n_jobs)
        f1 = [x["macro_f1"] for x in r]; rh = [x["recall_high"] for x in r]
        rows.append({"model": n, "macro_f1_mean": np.mean(f1), "macro_f1_sd": np.std(f1, ddof=1),
                     "recall_high_mean": np.mean(rh),
                     "recall_interm_mean": np.mean([x["recall_intermediate"] for x in r])})
        print(f"{n:13s} MacroF1={rows[-1]['macro_f1_mean']:.3f}±{rows[-1]['macro_f1_sd']:.3f}  recallCao={np.mean(rh):.3f}", flush=True)
        pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).to_csv(summary_path, index=False)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    res = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    print("\n", res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
