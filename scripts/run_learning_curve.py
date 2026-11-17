"""Learning curve trên nhóm 600 bệnh nhân dành cho mô hình lượng tử.

Dữ liệu huấn luyện lấy dần từ nhóm 600 người; tập đánh giá cố định là phần còn lại của Development.
Thêm --tune để tinh chỉnh siêu tham số bằng inner 3-fold như CV và Final Test.
Phân tích chính dùng scripts/run_learning_curve_full.py trên toàn bộ Development.
Chạy: python -m scripts.run_learning_curve --sizes 50 100 200 400 600 --repeats 5 --tune
Kết quả: results/learning_curve/metrics.csv (không tinh chỉnh) hoặc metrics_tuned.csv (--tune)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.dataset import load_dataset, split_ids
from src.evaluation import classif_zoo, make_pipeline, tuned_predict

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=int, default=[50, 100, 200, 400, 600])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--models", nargs="*", default=["LR", "SVM", "HGB", "QSVC", "QSVC-product"])
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args()

    df = load_dataset(ROOT)
    dev = set(split_ids(ROOT, "development").tolist())
    pool = set(pd.read_csv(ROOT / "data/splits/quantum_dev_pool.csv").raw_row_index.tolist())
    pool_ids = np.array(sorted(pool & dev)); val_ids = np.array(sorted(dev - pool))
    Xval = df.loc[val_ids]; yval = df.loc[val_ids, "y"].to_numpy()
    zoo = classif_zoo()
    rows = []
    for rep_seed in range(a.repeats):
        for n in a.sizes:
            if n >= len(pool_ids):
                tr = pool_ids
            else:
                tr, _ = train_test_split(pool_ids, train_size=n, random_state=rep_seed,
                                         stratify=df.loc[pool_ids, "y"])
            Xtr = df.loc[tr]; ytr = df.loc[tr, "y"].to_numpy()
            for m in a.models:
                model, grid, kind = zoo[m]
                if a.tune:
                    pred, bp = tuned_predict(Xtr, Xval, model, grid, kind, "domain", a.k, n_jobs=a.n_jobs)
                else:
                    pred, bp = make_pipeline(model, kind, "domain", a.k).fit(Xtr, ytr).predict(Xval), {}
                rows.append({"model": m, "train_n": len(tr), "repeat": rep_seed,
                             "macro_f1": f1_score(yval, pred, average="macro"), "best_params": str(bp)})
            print(f"seed {rep_seed} n={len(tr)} xong", flush=True)
    out = ROOT / "results" / "learning_curve"; out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    summ = res.groupby(["model", "train_n"])["macro_f1"].agg(["mean", "std"]).round(3)
    suffix = "_tuned" if a.tune else ""
    res.to_csv(out / f"metrics{suffix}.csv", index=False)
    summ.to_csv(out / f"summary{suffix}.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
