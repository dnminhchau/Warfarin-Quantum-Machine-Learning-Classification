"""Đánh giá trên Final Test (1.082 bệnh nhân).

Mỗi mô hình được tinh chỉnh bằng inner CV trên Development, huấn luyện lại trên toàn bộ
Development rồi dự đoán Final Test; sau đó so sánh các cặp mô hình bằng paired bootstrap.
Chỉ chạy khi pipeline đã cố định. Script dừng nếu đã có kết quả để tránh mở lại tập kiểm thử.
Chạy: python -m scripts.evaluate_final_test --k 4 --n-jobs -1
Kết quả: results/final_test/metrics.csv và bootstrap.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import load_dataset, split_ids
from src.evaluation import (classif_zoo, classification_metrics, paired_bootstrap,
                                tuned_predict)

ROOT = Path(__file__).resolve().parent.parent

# Các cặp so sánh bằng paired bootstrap, khai báo trước khi đánh giá
PAIRS = [("LR", "QSVC-product"), ("QSVC-product", "QSVC"), ("QSVC-product", "SVM")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rep", default="domain")
    ap.add_argument("--models", nargs="*", default=["LR", "SVM", "MLP", "HGB", "QSVC", "QSVC-product", "VQC", "Dummy"])
    ap.add_argument("--vqc-seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()

    if (ROOT / "results/final_test/metrics.csv").exists():
        raise FileExistsError("Final Test results already exist in results/final_test; do not rerun a completed evaluation.")

    df = load_dataset(ROOT)
    dev = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]
    test = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "final_test"))]
    yte = test["y"].to_numpy()
    print(f"Train Development n={len(dev)} | Final Test n={len(test)} (Thấp/Vừa/Cao={np.bincount(yte).tolist()})\n")

    zoo = classif_zoo(a.vqc_seed); preds = {}; rows = []
    # Công thức IWPC, không tinh chỉnh
    from src.baselines import iwpc_predict_class
    preds["IWPC"] = iwpc_predict_class(test)
    rows.append({"model": "IWPC", **classification_metrics(yte, preds["IWPC"]), "best_params": {}})
    for n in a.models:
        model, grid, kind = zoo[n]
        p, bp = tuned_predict(dev, test, model, grid, kind, a.rep, a.k, n_jobs=a.n_jobs)
        preds[n] = p
        rows.append({"model": n, **classification_metrics(yte, p), "best_params": bp})
        print(f"{n:13s} MacroF1={rows[-1]['macro_f1']:.3f}  recallCao={rows[-1]['recall_high']:.3f}  tune={bp}", flush=True)

    out = ROOT / "results" / "final_test"; out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    res_save = res.copy(); res_save["best_params"] = res_save["best_params"].map(json.dumps)
    res_save.to_csv(out / "metrics.csv", index=False)

    boot = {}
    for aa, bb in PAIRS:
        if aa in preds and bb in preds:
            boot[f"{aa}_vs_{bb}"] = paired_bootstrap(yte, preds[aa], preds[bb])
    (out / "bootstrap.json").write_text(json.dumps(boot, indent=2, ensure_ascii=False))
    print("\n", res[["model", "macro_f1", "accuracy", "recall_low", "recall_intermediate",
                     "recall_high", "qwk", "severe_error_low_high"]].round(3).to_string(index=False))
    print("\nPaired bootstrap (Macro-F1):")
    for k, v in boot.items():
        print(f"  {k}: {v['mean_diff']:+.3f}  CI=[{v['ci_low']:+.3f},{v['ci_high']:+.3f}]  "
              f"{'KHÁC BIỆT có ý nghĩa' if v['significant'] else 'chồng 0 (không khác biệt)'}")


if __name__ == "__main__":
    main()
