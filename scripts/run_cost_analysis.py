"""Chi phí tính toán theo số đặc trưng/qubit (k = 2, 4, 6) và theo số mẫu huấn luyện.

Dùng fold 0 của CV phân tầng theo trung tâm (seed 42) trên Development, với tham số đã chọn ở Final Test
(không tinh chỉnh lại). Đo thời gian huấn luyện và dự đoán trên một luồng CPU, kèm thống kê mạch lượng tử.
VQC ở các k trong --vqc-estimate-ks chỉ chạy --vqc-estimate-iters vòng tối ưu; thời gian huấn luyện đầy đủ
được ước tính bằng (giây mỗi vòng) x (số vòng tối đa) và đánh dấu trong cột estimated.
Nên chạy với OMP_NUM_THREADS=1 để thời gian các mô hình so sánh được với nhau.
Chạy: python -m scripts.run_cost_analysis
Kết quả: results/cost/cost_by_qubits.csv, cost_by_train_size.csv, environment.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

from src.dataset import load_dataset, site_labels, split_ids
from src.evaluation import classification_metrics, make_pipeline
from src.quantum_models import QuantumKernelSVC, VQCClassifier, make_feature_map

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cost"
VQC_MAXITER = 250
# Tham số được chọn ở Final Test (results/final_test/metrics.csv, metrics_additional.csv)
MODELS = {
    "LR": (LogisticRegression(C=100, max_iter=5000, class_weight="balanced"), "classical"),
    "SVM": (SVC(C=10, gamma=0.01, class_weight="balanced"), "classical"),
    "QSVC": (QuantumKernelSVC(feature_map="zz", bandwidth=0.05, C=10), "quantum"),
    "QSVC-product": (QuantumKernelSVC(feature_map="product", bandwidth=0.1, C=10), "quantum"),
    "VQC": (VQCClassifier(layers=2, reps=3, bandwidth=0.5, maxiter=VQC_MAXITER, seed=0), "quantum"),
}


def circuit_info(name: str, k: int, fitted) -> dict:
    """Số qubit, độ sâu, số CNOT và số tham số huấn luyện của mạch (trước transpile)."""
    if name.startswith("QSVC"):
        fm = make_feature_map(k, "zz" if name == "QSVC" else "product", 2, "linear").decompose(reps=3)
        return {"qubits": k, "circuit_depth": fm.depth(), "cnot": int(fm.count_ops().get("cx", 0)),
                "quantum_params": 0, "statevector_dim": 2 ** k}
    if name == "VQC":
        s = fitted.circuit_stats()
        return {"qubits": k, "circuit_depth": s["depth"], "cnot": s["cx"],
                "quantum_params": s["trainable_quantum_params"], "statevector_dim": 2 ** k}
    return {}


def measure(name, model, kind, k, Xtr, ytr, Xva, yva, repeats, maxiter=None):
    est = clone(model)
    if maxiter is not None:
        est.set_params(maxiter=maxiter)
    fit_t, pred_t = [], []
    for _ in range(repeats):
        pipe = make_pipeline(est, kind, "domain", k)
        t0 = time.perf_counter(); pipe.fit(Xtr, ytr); fit_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); pred = pipe.predict(Xva); pred_t.append(time.perf_counter() - t0)
    fitted = pipe.named_steps["model"]
    row = {"model": name, "k": k, "n_train": len(Xtr), "n_val": len(Xva),
           "fit_seconds": float(np.median(fit_t)), "predict_seconds": float(np.median(pred_t)),
           "macro_f1": classification_metrics(yva, pred)["macro_f1"], "estimated": False,
           **circuit_info(name, k, fitted)}
    if name == "VQC":
        per_iter = row["fit_seconds"] / max(fitted.n_iter_, 1)
        row.update({"vqc_iterations": fitted.n_iter_, "seconds_per_iteration": per_iter})
        if maxiter is not None:
            row.update({"fit_seconds": per_iter * VQC_MAXITER, "estimated": True, "macro_f1": np.nan})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", nargs="*", type=int, default=[2, 4, 6])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--vqc-estimate-ks", nargs="*", type=int, default=[6])
    ap.add_argument("--vqc-estimate-iters", type=int, default=20)
    ap.add_argument("--sizes", nargs="*", type=int, default=[500, 1000, 2000])
    a = ap.parse_args()

    df = load_dataset(ROOT)
    dev = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]
    y = dev["y"].to_numpy()
    tr, va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(dev, site_labels(dev)))
    Xtr, ytr, Xva, yva = dev.iloc[tr], y[tr], dev.iloc[va], y[va]
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in a.ks:
        for name, (model, kind) in MODELS.items():
            if name == "VQC":
                est = k in a.vqc_estimate_ks
                rows.append(measure(name, model, kind, k, Xtr, ytr, Xva, yva, 1,
                                    a.vqc_estimate_iters if est else None))
            else:
                rows.append(measure(name, model, kind, k, Xtr, ytr, Xva, yva, a.repeats))
            r = rows[-1]
            print(f"k={k} {name:13s} fit={r['fit_seconds']:9.2f}s{' (ước tính)' if r['estimated'] else ''}"
                  f"  predict={r['predict_seconds']:.3f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "cost_by_qubits.csv", index=False)

    rows_n = []
    for n in a.sizes + [len(tr)]:
        sub = tr if n >= len(tr) else train_test_split(tr, train_size=n, random_state=0, stratify=y[tr])[0]
        for name in ["LR", "SVM", "QSVC", "QSVC-product"]:
            model, kind = MODELS[name]
            rows_n.append(measure(name, model, kind, 4, dev.iloc[sub], y[sub], Xva, yva, a.repeats))
            print(f"n={len(sub)} {name:13s} fit={rows_n[-1]['fit_seconds']:.2f}s", flush=True)
        pd.DataFrame(rows_n).to_csv(OUT / "cost_by_train_size.csv", index=False)

    (OUT / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "fold": "fold 0 of StratifiedKFold(5, shuffle=True, random_state=42) on Project Site",
        "note": "Thời gian phụ thuộc phần cứng; chỉ nên so sánh tương đối giữa các mô hình trong cùng lần chạy.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
