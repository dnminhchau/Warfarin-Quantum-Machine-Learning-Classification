"""Đánh giá Final Test bổ sung cho các mô hình chưa có trong results/final_test/metrics.csv.

Không ghi đè kết quả đã có. Pipeline và lưới tham số giữ nguyên như evaluate_final_test.py.
Lần chạy này được ghi lại trong docs/final_configuration.md.
Chạy: python -m scripts.evaluate_final_test_additional --models MLP HGB VQC
Kết quả: results/final_test/metrics_additional.csv và additional_run.json (danh sách các lần chạy)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import load_dataset, split_ids
from src.evaluation import classif_zoo, classification_metrics, tuned_predict

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rep", default="domain")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--vqc-seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args()

    out = ROOT / "results" / "final_test"
    done = set(pd.read_csv(out / "metrics.csv").model)
    path = out / "metrics_additional.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done |= {r["model"] for r in rows}

    df = load_dataset(ROOT)
    dev = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]
    test = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "final_test"))]
    yte = test["y"].to_numpy()
    zoo = classif_zoo(a.vqc_seed)
    for n in a.models:
        if n in done:
            print(f"{n}: đã có kết quả Final Test, bỏ qua"); continue
        model, grid, kind = zoo[n]
        p, bp = tuned_predict(dev, test, model, grid, kind, a.rep, a.k, n_jobs=a.n_jobs)
        rows.append({"model": n, **classification_metrics(yte, p), "best_params": json.dumps(bp)})
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{n:6s} MacroF1={rows[-1]['macro_f1']:.3f} recallCao={rows[-1]['recall_high']:.3f} tune={bp}", flush=True)
    log = out / "additional_run.json"
    runs = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    runs = runs if isinstance(runs, list) else [runs]
    runs.append({
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": a.models, "vqc_seed": a.vqc_seed, "rep": a.rep, "k": a.k,
        "note": "Chỉ thêm mô hình chưa đánh giá; không thay đổi pipeline hay kết quả của các mô hình đã có.",
    })
    log.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
