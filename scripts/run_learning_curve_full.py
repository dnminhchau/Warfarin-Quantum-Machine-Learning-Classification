"""Learning curve trên toàn bộ Development (4.328 bệnh nhân), có tinh chỉnh siêu tham số.

Dùng cùng 5 fold phân tầng theo trung tâm như run_classification_cv (seed 42). Trong mỗi fold,
lấy ngẫu nhiên n người từ phần huấn luyện (phân tầng theo nhãn, seed = số thứ tự fold), tinh chỉnh
bằng inner 3-fold và đánh giá trên fold kiểm định. Mốc full dùng toàn bộ phần huấn luyện, trùng với
kết quả trong results/classification_cv. Không dùng Final Test.
Chạy: python -m scripts.run_learning_curve_full --sizes 50 100 200 400 600 1000 2000 full
Kết quả: results/learning_curve_full/metrics.csv và summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.dataset import load_dataset, site_labels, split_ids
from src.evaluation import classif_zoo, classification_metrics, make_pipeline

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", default=["50", "100", "200", "400", "600", "1000", "2000", "full"])
    ap.add_argument("--models", nargs="*", default=["LR", "SVM", "HGB", "QSVC", "QSVC-product"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()

    df = load_dataset(ROOT)
    dev = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]
    y = dev["y"].to_numpy()
    outer = list(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(dev, site_labels(dev)))
    out = ROOT / "results" / "learning_curve_full"; out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["model"], str(r["size"]), int(r["fold"])) for r in rows}
    zoo = classif_zoo()
    for m in a.models:
        model, grid, kind = zoo[m]
        for s in a.sizes:
            for f, (tr, va) in enumerate(outer):
                if (m, s, f) in done:
                    continue
                if s == "full" or int(s) >= len(tr):
                    sub = tr
                else:
                    sub, _ = train_test_split(tr, train_size=int(s), random_state=f, stratify=y[tr])
                pipe = make_pipeline(model, kind, "domain", a.k)
                if grid:
                    gs = GridSearchCV(pipe, grid, scoring="f1_macro", n_jobs=a.n_jobs,
                                      cv=StratifiedKFold(3, shuffle=True, random_state=a.seed))
                    est = gs.fit(dev.iloc[sub], y[sub]); bp = gs.best_params_
                else:
                    est = pipe.fit(dev.iloc[sub], y[sub]); bp = {}
                rows.append({"model": m, "size": s, "train_n": len(sub), "fold": f,
                             **classification_metrics(y[va], est.predict(dev.iloc[va])), "best_params": str(bp)})
                pd.DataFrame(rows).to_csv(path, index=False)
            last = pd.DataFrame(rows).query("model == @m and size == @s")
            print(f"{m:13s} n={s:>5s}  MacroF1={last.macro_f1.mean():.3f}±{last.macro_f1.std():.3f}", flush=True)
    res = pd.DataFrame(rows)
    summ = (res.groupby(["model", "size"], sort=False)
            .agg(train_n=("train_n", "mean"), macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                 recall_high_mean=("recall_high", "mean")).round(4).reset_index())
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
