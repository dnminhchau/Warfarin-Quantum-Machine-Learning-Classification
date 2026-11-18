"""CV lặp lại phân tầng theo trung tâm: kiểm định Nadeau–Bengio và độ ổn định của VQC theo seed.

Lần lặp r dùng StratifiedKFold(5, shuffle=True, random_state=42 + r) trên mã trung tâm; lần lặp 0 trùng
với scripts/run_classification_cv.py. Siêu tham số chọn bằng inner 3-fold (seed 42). Không dùng Final Test.
Kết quả được lưu sau mỗi fold; chạy lại sẽ bỏ qua các fold đã có.
Chạy:
    python -m scripts.run_repeated_cv --repeats 0 1 2 3 4 --models LR SVM QSVC QSVC-product VQC
    python -m scripts.run_repeated_cv --repeats 0 --models VQC --vqc-seeds 1 2 3 4 --tag vqc_seeds
    python -m scripts.run_repeated_cv --summarize
Kết quả: results/repeated_cv/fold_metrics_<tag>.csv, summary.csv, nadeau_bengio.csv, vqc_seeds.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.dataset import load_dataset, site_labels, split_ids
from src.evaluation import classif_zoo, classification_metrics, make_pipeline

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "repeated_cv"
# Các cặp so sánh khai báo trong docs/protocol.md, thêm cặp LR – QSVC-product của kết quả chính
PAIRS = [("QSVC", "SVM"), ("VQC", "LR"), ("QSVC", "QSVC-product"), ("LR", "QSVC-product")]


def run(a):
    df = load_dataset(ROOT)
    dev = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]
    y, sites = dev["y"].to_numpy(), site_labels(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"fold_metrics_{a.tag}.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["model"], int(r["vqc_seed"]), int(r["repeat"]), int(r["fold"])) for r in rows}
    for m in a.models:
        seeds = a.vqc_seeds if m == "VQC" else [0]
        for s in seeds:
            model, grid, kind = classif_zoo(s)[m]
            for r in a.repeats:
                outer = StratifiedKFold(5, shuffle=True, random_state=42 + r).split(dev, sites)
                for f, (tr, va) in enumerate(outer):
                    if (m, s, r, f) in done:
                        continue
                    gs = GridSearchCV(make_pipeline(model, kind, "domain", 4), grid, scoring="f1_macro",
                                      cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=a.n_jobs)
                    gs.fit(dev.iloc[tr], y[tr])
                    rows.append({"model": m, "vqc_seed": s, "repeat": r, "fold": f,
                                 "n_train": len(tr), "n_val": len(va),
                                 **classification_metrics(y[va], gs.predict(dev.iloc[va])),
                                 "best_params": str(gs.best_params_)})
                    pd.DataFrame(rows).to_csv(path, index=False)
                cur = pd.DataFrame(rows).query("model == @m and vqc_seed == @s and repeat == @r")
                print(f"{m:13s} seed={s} repeat={r}  MacroF1={cur.macro_f1.mean():.3f}", flush=True)


def nadeau_bengio(d: np.ndarray, n_train: float, n_val: float) -> tuple[float, float]:
    """Kiểm định t hiệu chỉnh cho CV lặp lại (Nadeau & Bengio, 2003); trả về (t, p hai phía)."""
    j = len(d)
    var = np.var(d, ddof=1) * (1 / j + n_val / n_train)
    t = d.mean() / np.sqrt(var)
    return float(t), float(2 * stats.t.sf(abs(t), df=j - 1))


def summarize():
    res = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("fold_metrics_*.csv"))], ignore_index=True)
    main = res[res.vqc_seed == 0]
    summary = (main.groupby("model")
               .agg(n_folds=("macro_f1", "size"), n_repeats=("repeat", "nunique"),
                    macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                    recall_high_mean=("recall_high", "mean"), qwk_mean=("qwk", "mean"))
               .sort_values("macro_f1_mean", ascending=False).round(4).reset_index())
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False), "\n")

    tests = []
    for a_, b_ in PAIRS:
        A = main[main.model == a_].set_index(["repeat", "fold"])
        B = main[main.model == b_].set_index(["repeat", "fold"])
        idx = A.index.intersection(B.index)
        if len(idx) < 2:
            continue
        d = (A.loc[idx, "macro_f1"] - B.loc[idx, "macro_f1"]).to_numpy()
        t, p = nadeau_bengio(d, A.loc[idx, "n_train"].mean(), A.loc[idx, "n_val"].mean())
        tests.append({"pair": f"{a_} - {b_}", "n_folds": len(idx), "mean_diff": round(d.mean(), 4),
                      "t": round(t, 3), "p_value": round(p, 4), "significant_0.05": p < 0.05})
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / "nadeau_bengio.csv", index=False)
    print(tests.to_string(index=False), "\n")

    vqc = res[(res.model == "VQC") & (res.repeat == 0)]
    if not vqc.empty:
        per_seed = vqc.groupby("vqc_seed").agg(macro_f1_mean=("macro_f1", "mean"),
                                               recall_high_mean=("recall_high", "mean"),
                                               n_folds=("macro_f1", "size")).round(4).reset_index()
        per_seed.to_csv(OUT / "vqc_seeds.csv", index=False)
        print(per_seed.to_string(index=False))
        print(f"VQC qua {len(per_seed)} seed: {per_seed.macro_f1_mean.mean():.3f} ± {per_seed.macro_f1_mean.std():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="*", default=["LR", "SVM", "QSVC", "QSVC-product", "VQC"])
    ap.add_argument("--vqc-seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a)


if __name__ == "__main__":
    main()
