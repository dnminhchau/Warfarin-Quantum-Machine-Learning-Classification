"""Tái tạo dự đoán Final Test với cấu hình đã chốt, lưu dự đoán từng bệnh nhân và kiểm định các cặp khai báo trước.

Pipeline, lưới tham số và seed giữ nguyên như evaluate_final_test.py. Chỉ số của từng mô hình được đối chiếu
với results/final_test/metrics.csv và metrics_additional.csv; script dừng nếu có sai khác. Sau đó tính paired
bootstrap cho các cặp khai báo trong docs/protocol.md và ma trận nhầm lẫn của từng mô hình.
Chạy: python -m scripts.evaluate_final_test_predictions
Kết quả: results/final_test/predictions.csv, bootstrap_predeclared.json, confusion_matrices.csv,
predictions_run.json (danh sách các lần chạy)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.baselines import iwpc_predict_class
from src.dataset import CLASS_ORDER, load_dataset, split_ids
from src.evaluation import classif_zoo, classification_metrics, paired_bootstrap, tuned_predict

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "final_test"
MODELS = ["LR", "SVM", "MLP", "MLP-balanced", "HGB", "QSVC", "QSVC-product", "VQC"]
PREDECLARED = [("QSVC", "SVM"), ("VQC", "LR"), ("QSVC", "QSVC-product")]
EXISTING = [("LR", "QSVC-product"), ("QSVC-product", "QSVC"), ("QSVC-product", "SVM")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()

    reference = pd.concat([pd.read_csv(OUT / "metrics.csv"), pd.read_csv(OUT / "metrics_additional.csv")])
    reference = reference.set_index("model")
    path = OUT / "predictions.csv"
    df = load_dataset(ROOT)
    dev = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]
    test = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "final_test"))]
    yte = test["y"].to_numpy()
    preds = pd.read_csv(path, index_col="raw_row_index") if path.exists() else pd.DataFrame(
        {"y_true": yte}, index=pd.Index(test.index, name="raw_row_index"))
    preds["IWPC"] = iwpc_predict_class(test)

    zoo = classif_zoo(0)
    for m in MODELS:
        if m not in preds:
            model, grid, kind = zoo[m]
            preds[m], bp = tuned_predict(dev, test, model, grid, kind, "domain", 4, n_jobs=a.n_jobs)
            preds.to_csv(path)
            print(f"{m:13s} tune={bp}", flush=True)

    checks = []
    for m in MODELS + ["IWPC"]:
        got = classification_metrics(yte, preds[m].to_numpy())
        diff = max(abs(got[k] - reference.loc[m, k]) for k in got)
        checks.append({"model": m, "macro_f1": round(got["macro_f1"], 4), "max_abs_diff_vs_saved": diff})
    checks = pd.DataFrame(checks)
    print(checks.to_string(index=False))
    if (checks.max_abs_diff_vs_saved > 1e-9).any():
        raise SystemExit("Dự đoán không khớp với kết quả đã lưu; dừng, không tính bootstrap.")

    boot = {}
    for group, pairs in [("predeclared", PREDECLARED), ("existing_recomputed", EXISTING)]:
        boot[group] = {f"{x}_vs_{y}": paired_bootstrap(yte, preds[x].to_numpy(), preds[y].to_numpy())
                       for x, y in pairs}
    saved = json.loads((OUT / "bootstrap.json").read_text(encoding="utf-8"))
    boot["existing_matches_saved"] = all(
        abs(boot["existing_recomputed"][k]["mean_diff"] - v["mean_diff"]) < 1e-12 for k, v in saved.items())
    (OUT / "bootstrap_predeclared.json").write_text(json.dumps(boot, indent=2, ensure_ascii=False), encoding="utf-8")

    cms = []
    for m in MODELS + ["IWPC"]:
        cm = confusion_matrix(yte, preds[m].to_numpy(), labels=[0, 1, 2])
        for i, t in enumerate(CLASS_ORDER):
            cms.append({"model": m, "true": t, **{f"pred_{p}": int(cm[i, j]) for j, p in enumerate(CLASS_ORDER)}})
    pd.DataFrame(cms).to_csv(OUT / "confusion_matrices.csv", index=False)

    log = OUT / "predictions_run.json"
    runs = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    runs = runs if isinstance(runs, list) else [runs]
    runs.append({
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": MODELS + ["IWPC"],
        "metrics_reproduced": True,
        "note": "Tái tạo dự đoán với cấu hình đã chốt (chỉ số khớp kết quả đã lưu) để tính bootstrap cho "
                "các cặp khai báo trước và ma trận nhầm lẫn; mô hình chưa có dự đoán được tinh chỉnh và dự đoán "
                "theo cùng pipeline. Lịch sử sử dụng Final Test: docs/final_configuration.md.",
    })
    log.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in boot["predeclared"].items():
        print(f"{k}: {v['mean_diff']:+.4f}  CI=[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]  "
              f"{'có ý nghĩa' if v['significant'] else 'không có ý nghĩa'}")
    print("3 cặp cũ tính lại khớp bootstrap.json:", boot["existing_matches_saved"])


if __name__ == "__main__":
    main()
