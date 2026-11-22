"""Khảo sát số qubit và feature map.

--part qubits: số đặc trưng/qubit k = 2, 4, 6 cho LR, SVM, MLP-balanced, QSVC, QSVC-product, VQC
               (CV phân tầng theo trung tâm, có tinh chỉnh). Kết quả mới được ghép vào file đã có.
--part maps:   với k = 4, so sánh các feature map của QSVC (zz linear/full, pauli_yzz, product,
               angle_ry) và đo mức tập trung của hàm nhân.
Chạy: python -m scripts.run_qubit_ablation --part qubits --ks 2 4 6 --models MLP-balanced
Kết quả: results/ablation/qubit_summary.csv và feature_map_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.dataset import load_dataset, site_labels, split_ids
from src.evaluation import classif_zoo, classification_metrics, cv_site_stratified
from src.preprocessing import build_representation
from src.quantum_models import QuantumKernelSVC

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["qubits", "maps", "both"], default="both")
    ap.add_argument("--ks", nargs="*", type=int, default=[2, 4, 6])
    ap.add_argument("--models", nargs="*", default=["LR", "SVM", "MLP-balanced", "QSVC", "QSVC-product", "VQC"])
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    df = load_dataset(ROOT)
    dev = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]
    zoo = classif_zoo()

    out = ROOT / "results" / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    if a.part in {"qubits", "both"}:
        # Khảo sát số qubit
        path = out / "qubit_summary.csv"
        rows = pd.read_csv(path).to_dict("records") if path.exists() else []
        for k in a.ks:
            for n in a.models:
                rows = [r for r in rows if not (r["k"] == k and r["model"] == n)]
                model, grid, kind = zoo[n]
                r = cv_site_stratified(dev, model, grid, kind, "domain", k, n_jobs=a.n_jobs)
                f1 = [x["macro_f1"] for x in r]
                rows.append({"k": k, "model": n, "macro_f1_mean": np.mean(f1), "macro_f1_sd": np.std(f1, ddof=1)})
                print(f"k={k} {n:13s} MacroF1={np.mean(f1):.3f}", flush=True)
                pd.DataFrame(rows).sort_values(["k", "model"]).to_csv(path, index=False)

    if a.part == "qubits":
        return

    # So sánh feature map tại k = 4, kèm trung bình phần tử ngoài đường chéo của hàm nhân
    y = dev["y"].to_numpy(); st = site_labels(dev); skf = StratifiedKFold(5, shuffle=True, random_state=42)
    fmaps = [("zz_linear", dict(feature_map="zz", entanglement="linear")),
             ("zz_full", dict(feature_map="zz", entanglement="full")),
             ("pauli_yzz", dict(feature_map="pauli_yzz")),
             ("product(no-entangle)", dict(feature_map="product")),
             ("angle_ry", dict(feature_map="angle_ry"))]
    frows = []
    for name, kw in fmaps:
        f1s, offd = [], []
        for tr, va in skf.split(dev, st):
            pre = Pipeline(build_representation("domain", 4, True, angle=True)).fit(dev.iloc[tr])
            m = QuantumKernelSVC(bandwidth=0.25, C=10, **kw).fit(pre.transform(dev.iloc[tr]), y[tr])
            f1s.append(classification_metrics(y[va], m.predict(pre.transform(dev.iloc[va])))["macro_f1"])
            offd.append(m.kernel_offdiag_mean_)
        frows.append({"feature_map": name, "macro_f1_mean": np.mean(f1s),
                      "kernel_offdiag_mean": np.mean(offd)})
        print(f"{name:22s} MacroF1={np.mean(f1s):.3f}  offdiag={np.mean(offd):.3f}", flush=True)
    pd.DataFrame(frows).to_csv(out / "feature_map_summary.csv", index=False)


if __name__ == "__main__":
    main()
