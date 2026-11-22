"""Ảnh hưởng của số lần đo (shots) và nhiễu phần cứng lên QSVC, QSVC-product và VQC.

Các kết quả chính của đồ án dùng statevector chính xác: xác suất được tính đúng tuyệt đối và mạch không có lỗi.
Script này mô phỏng hai điều xảy ra trên máy lượng tử thật:
- Shots: mỗi xác suất được ước lượng bằng tần suất trong N lần đo (N = 256 đến 65.536).
- Nhiễu: mạch transpile về cổng gốc (cx, rz, sx, x) trên 4 qubit nối thành hàng; nhiễu khử cực sau mỗi cổng
  và lỗi đọc, ở hai mức (xem NOISE_LEVELS và src/noise_simulation.py).

Thiết kế: CV 5-fold phân tầng theo trung tâm (seed 42) trên Development, giống CV chính. Tham số mô hình cố định
bằng tham số đã chọn ở Final Test, không tinh chỉnh lại. Không dùng Final Test.
QSVC: cả hàm nhân huấn luyện và hàm nhân dự đoán đều chịu shots/nhiễu; đường chéo hàm nhân huấn luyện đặt bằng 1
và ma trận được chiếu về nửa xác định dương, giống FidelityQuantumKernel của Qiskit.
VQC: huấn luyện bằng statevector chính xác như kết quả chính, chỉ phần dự đoán chịu shots/nhiễu
(huấn luyện VQC có nhiễu cần số mạch quá lớn, xem hardware_cost.csv).

Chạy: python -m scripts.run_noise_analysis
Kết quả: results/noise/noise_cv_folds.csv, noise_summary.csv, circuits.csv, hardware_cost.csv, settings.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.sparse.linalg import eigsh
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from src.dataset import load_dataset, site_labels, split_ids
from src.evaluation import classification_metrics, make_pipeline
from src.noise_simulation import (NoiseModel, Program, gate_counts, kernel_from, project_psd, sample_shots,
                                  transpile_line)
from src.quantum_models import QuantumKernelSVC, VQCClassifier, feature_states, make_feature_map

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "noise"
K_QUBITS = 4

# Tham số được chọn ở Final Test (results/final_test/metrics.csv, metrics_additional.csv)
QSVC_MODELS = {"QSVC": {"feature_map": "zz", "bandwidth": 0.05, "C": 10},
               "QSVC-product": {"feature_map": "product", "bandwidth": 0.1, "C": 10}}
VQC_PARAMS = {"layers": 2, "reps": 3, "bandwidth": 0.5, "maxiter": 250, "seed": 0}

# Mức thấp: lỗi cx 0,3% như ví dụ trong bài giảng W8, gần các máy IBM hiện nay.
# Mức cao: lỗi lớn hơn khoảng 3 lần, gần các thế hệ máy cũ hơn.
NOISE_LEVELS = {"none": None,
                "low": NoiseModel(p1=3e-4, p2=3e-3, readout=1e-2),
                "high": NoiseModel(p1=1e-3, p2=1e-2, readout=3e-2)}
# (mức nhiễu, số shots); shots = 0 nghĩa là xác suất chính xác
CONDITIONS = [("none", 0), ("none", 256), ("none", 1024), ("none", 4096), ("none", 16384), ("none", 65536),
              ("low", 0), ("low", 1024), ("high", 0), ("high", 1024)]
HW_SHOTS = 1000           # số shots mỗi mạch khi ước tính chi phí phần cứng (như ví dụ trong bài giảng W8)
EPS_CX = 0.003            # sai số mỗi CNOT trong công thức (1 - ε)^số CNOT của bài giảng W8


class CountingVQC(VQCClassifier):
    """VQC đếm số lần tính loss và gradient, để ước tính số mạch cần chạy trên phần cứng."""

    def _loss_grad(self, params, enc, Y, w):
        self.n_evals_ = getattr(self, "n_evals_", 0) + 1
        return super()._loss_grad(params, enc, Y, w)


def qsvc_fold(name, cfg, Atr, ytr, Ava, yva, rng_seed):
    """Macro-F1 của một QSVC trên một fold, với mọi điều kiện shots/nhiễu."""
    bw = cfg["bandwidth"]
    fmap = make_feature_map(K_QUBITS, cfg["feature_map"], 2, "linear")
    prog = Program(transpile_line(fmap))
    Xtr, Xva = bw * Atr, bw * Ava
    S_tr, S_va = feature_states(Xtr, fmap), feature_states(Xva, fmap)
    exact = {"tr": np.abs(S_tr.conj() @ S_tr.T) ** 2, "va": np.abs(S_va.conj() @ S_tr.T) ** 2}
    off = ~np.eye(len(Xtr), dtype=bool)

    probs = {"none": exact}
    for lvl in ("low", "high"):
        noise = NOISE_LEVELS[lvl]
        rho_tr, M_tr = prog.forward(Xtr, noise), prog.backward(Xtr, noise)
        probs[lvl] = {"tr": kernel_from(rho_tr, M_tr), "va": kernel_from(prog.forward(Xva, noise), M_tr)}
    # Kiểm tra: bộ mô phỏng không nhiễu cho đúng hàm nhân chính xác
    sub = slice(0, 50)
    k0 = kernel_from(prog.forward(Xtr[sub], None), prog.backward(Xtr[sub], None))
    assert np.allclose(k0, exact["tr"][sub, sub], atol=1e-10)

    # Hàm nhân "đúng" ở mỗi mức nhiễu khi đo vô hạn lần (đường chéo = 1) và phổ trị riêng của nó.
    # Hàm nhân tập trung nên chỉ vài hướng có trị riêng lớn.
    ref, eig_ref = {}, {}
    for lvl in probs:
        ref[lvl] = probs[lvl]["tr"].copy()
        np.fill_diagonal(ref[lvl], 1.0)
        eig_ref[lvl] = np.linalg.eigvalsh(ref[lvl])[::-1]
    # Nhiễu cổng (khi đo vô hạn lần) gần như chỉ co hàm nhân: K_nhiễu ≈ a·K + c (hồi quy trên phần tử ngoài chéo)
    affine = {}
    for lvl in ("low", "high"):
        a, c = np.polyfit(exact["tr"][off], probs[lvl]["tr"][off], 1)
        resid = float((probs[lvl]["tr"][off] - (a * exact["tr"][off] + c)).std())
        K_eq = a * exact["tr"] + c
        np.fill_diagonal(K_eq, 1.0)
        pred = SVC(kernel="precomputed", C=cfg["C"], class_weight="balanced").fit(K_eq, ytr).predict(
            a * exact["va"] + c)
        affine[lvl] = {"noise_kernel_slope": a, "noise_kernel_intercept": c, "noise_kernel_resid_std": resid,
                       "macro_f1_exact_kernel_shrunk": classification_metrics(yva, pred)["macro_f1"]}

    rows = []
    for ci, (lvl, shots) in enumerate(CONDITIONS):
        rng = np.random.default_rng([*rng_seed, ci])
        p_tr, p_va = probs[lvl]["tr"], probs[lvl]["va"]
        extra = {}
        if lvl == "none" and shots == 0:
            K_tr, K_va = p_tr, p_va
        else:
            iu = np.triu_indices(len(Xtr), 1)
            upper = sample_shots(p_tr[iu], shots, rng) if shots else p_tr[iu]
            K_tr = np.eye(len(Xtr))
            K_tr[iu] = upper
            K_tr = K_tr + np.triu(K_tr, 1).T
            if shots:
                # Sai số do shots của cả ma trận (chuẩn phổ) và số trị riêng của hàm nhân đo vô hạn lần
                # lớn hơn sai số đó (số hướng thông tin còn phân biệt được)
                err = float(np.abs(eigsh(K_tr - ref[lvl], k=1, which="LM", return_eigenvectors=False)[0]))
                extra = {"shot_error_norm": err, "eigs_above_shot_error": int((eig_ref[lvl] > err).sum())}
            K_tr = project_psd(K_tr)
            K_va = sample_shots(p_va, shots, rng) if shots else p_va
            if shots == 0:
                extra |= affine[lvl]
        pred = SVC(kernel="precomputed", C=cfg["C"], class_weight="balanced").fit(K_tr, ytr).predict(K_va)
        rows.append({"model": name, "noise": lvl, "shots": shots, **classification_metrics(yva, pred), **extra,
                     "kernel_offdiag_mean": float(p_tr[off].mean()),
                     "kernel_offdiag_std": float(p_tr[off].std()),
                     "kernel_eig_5": float(eig_ref[lvl][4]), "kernel_eig_10": float(eig_ref[lvl][9]),
                     "mean_abs_prob_error": float(np.abs(K_va - exact["va"]).mean())})
    return rows, prog


def vqc_fold(Atr, ytr, Ava, yva, rng_seed):
    """VQC huấn luyện chính xác; dự đoán với mọi điều kiện shots/nhiễu."""
    model = CountingVQC(**VQC_PARAMS).fit(Atr, ytr)
    theta, W, b = model._split(model.params_)
    E_exact = model._expect_z(model._encode(Ava), model._blocks(theta))
    circ = QuantumCircuit(K_QUBITS)
    for l in range(model.layers):
        circ.compose(model.fmap_, inplace=True)
        circ.compose(model.ansatz_.assign_parameters(theta[l * model.p_block_:(l + 1) * model.p_block_]),
                     inplace=True)
    prog = Program(transpile_line(circ))
    Xva = model.bandwidth * Ava
    p1 = {lvl: prog.prob_one(Xva, NOISE_LEVELS[lvl]) for lvl in NOISE_LEVELS}
    assert np.allclose(1 - 2 * p1["none"], E_exact, atol=1e-10)

    rows = []
    for ci, (lvl, shots) in enumerate(CONDITIONS):
        rng = np.random.default_rng([*rng_seed, ci])
        p = sample_shots(p1[lvl], shots, rng) if shots else p1[lvl]
        E = 1 - 2 * p
        pred = np.argmax(E @ W.T + b, axis=1)
        rows.append({"model": "VQC", "noise": lvl, "shots": shots, **classification_metrics(yva, pred),
                     "mean_abs_prob_error": float(np.abs(p - p1["none"]).mean())})
    info = {"iterations": model.n_iter_, "loss_grad_evals": model.n_evals_,
            "trainable_quantum_params": model.n_theta_}
    return rows, prog, info


def hardware_cost(circuit_rows, vqc_info, n_train, n_test):
    """Số mạch và số lần chạy (shots) nếu huấn luyện trên toàn bộ Development và dự đoán Final Test."""
    evals = float(np.mean([v["loss_grad_evals"] for v in vqc_info]))
    P = vqc_info[0]["trainable_quantum_params"]
    rows = []
    for name in ["QSVC", "QSVC-product"]:
        fit = n_train * (n_train - 1) // 2
        pred = n_test * n_train
        rows.append({"model": name, "train_circuits": fit, "predict_circuits": pred,
                     "note": "Mỗi cặp bệnh nhân một mạch; chưa tính tinh chỉnh tham số"})
    fit = int(round(evals * (2 * P + 1) * n_train))
    rows.append({"model": "VQC", "train_circuits": fit, "predict_circuits": n_test,
                 "note": f"Mỗi lần tính gradient: 1 + 2 x {P} mạch cho mỗi mẫu (parameter-shift); "
                         f"trung bình {evals:.0f} lần tính trên 5 fold; chưa tính tinh chỉnh tham số"})
    df = pd.DataFrame(rows)
    df["total_circuits"] = df.train_circuits + df.predict_circuits
    df["shots_per_circuit"] = HW_SHOTS
    df["total_shots"] = df.total_circuits * HW_SHOTS
    circ = pd.DataFrame(circuit_rows).set_index("model")
    df["cx_per_circuit"] = df.model.map(circ["cx"])
    return df


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    base = folds[(folds.noise == "none") & (folds.shots == 0)][["model", "fold", "macro_f1"]]
    m = folds.merge(base.rename(columns={"macro_f1": "exact_f1"}), on=["model", "fold"])
    m["diff_vs_exact"] = m.macro_f1 - m.exact_f1
    g = m.groupby(["model", "noise", "shots"], sort=False)
    out = g.agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                recall_high_mean=("recall_high", "mean"),
                diff_vs_exact_mean=("diff_vs_exact", "mean"), diff_vs_exact_std=("diff_vs_exact", "std"),
                mean_abs_prob_error=("mean_abs_prob_error", "mean"),
                kernel_offdiag_mean=("kernel_offdiag_mean", "mean"),
                shot_error_norm=("shot_error_norm", "mean"),
                eigs_above_shot_error=("eigs_above_shot_error", "mean"),
                noise_kernel_slope=("noise_kernel_slope", "mean"),
                macro_f1_exact_kernel_shrunk=("macro_f1_exact_kernel_shrunk", "mean")).reset_index()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_dataset(ROOT)
    dev = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]
    y = dev["y"].to_numpy()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(dev, site_labels(dev)))

    rows, vqc_info, progs = [], [], {}
    for f in a.folds:
        t0 = time.perf_counter()
        tr, va = splits[f]
        pre = make_pipeline(QuantumKernelSVC(), "quantum", "domain", K_QUBITS)[:-1]
        Atr = pre.fit_transform(dev.iloc[tr], y[tr])
        Ava = pre.transform(dev.iloc[va])
        for mi, (name, cfg) in enumerate(QSVC_MODELS.items()):
            r, progs[name] = qsvc_fold(name, cfg, Atr, y[tr], Ava, y[va], [f, mi])
            rows += [{"fold": f, **x} for x in r]
        r, progs["VQC"], info = vqc_fold(Atr, y[tr], Ava, y[va], [f, 9])
        rows += [{"fold": f, **x} for x in r]
        vqc_info.append({"fold": f, **info})
        folds = pd.DataFrame(rows)
        folds.to_csv(OUT / "noise_cv_folds.csv", index=False)
        print(f"fold {f}: {time.perf_counter() - t0:.0f}s", flush=True)
        print(folds[folds.fold == f][["model", "noise", "shots", "macro_f1"]].to_string(index=False), flush=True)

    summary = summarize(folds)
    summary.to_csv(OUT / "noise_summary.csv", index=False)

    circuit_rows = []
    for name, prog in progs.items():
        c = gate_counts(prog.tc)
        if name.startswith("QSVC"):   # mạch hàm nhân: U(x_i) rồi U(x_j)†, mô phỏng như hai mạch nối tiếp
            c = {key: 2 * v for key, v in c.items()}
            fmap = make_feature_map(K_QUBITS, QSVC_MODELS[name]["feature_map"], 2, "linear")
            joint = transpile_line(fmap.compose(fmap.assign_parameters(list(ParameterVector("y", K_QUBITS))).inverse()))
            c["cx_transpiled_jointly"] = gate_counts(joint)["cx"]
            kind = "U(x_i) rồi U(x_j)†"
        else:
            kind = "2 lớp: ZZ feature map + RealAmplitudes (reps 3)"
        clean = (1 - EPS_CX) ** c["cx"]
        circuit_rows.append({"model": name, "circuit": kind, "qubits": K_QUBITS, **c,
                             "clean_run_prob_cx_0.3pct": round(clean, 4)})
    pd.DataFrame(circuit_rows).to_csv(OUT / "circuits.csv", index=False)

    cost = hardware_cost(circuit_rows, vqc_info, len(dev), len(split_ids(ROOT, "final_test")))
    cost.to_csv(OUT / "hardware_cost.csv", index=False)

    (OUT / "settings.json").write_text(json.dumps({
        "design": "CV 5-fold phân tầng theo Project Site (seed 42) trên Development; không dùng Final Test",
        "qsvc_params": QSVC_MODELS, "vqc_params": VQC_PARAMS,
        "noise_levels": {k: (v.as_dict() if v else None) for k, v in NOISE_LEVELS.items()},
        "conditions": [{"noise": n, "shots": s} for n, s in CONDITIONS],
        "transpile": "basis cx, rz, sx, x; coupling map dạng hàng 4 qubit; optimization_level 1; rz không lỗi",
        "qsvc_kernel": "đường chéo = 1, ước lượng tam giác trên rồi lấy đối xứng, chiếu về nửa xác định dương",
        "vqc": "huấn luyện statevector chính xác; shots/nhiễu chỉ áp dụng khi dự đoán",
        "vqc_training_info": vqc_info,
        "hardware_cost_shots_per_circuit": HW_SHOTS,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(pd.DataFrame(circuit_rows).to_string(index=False))
    print(cost.to_string(index=False))


if __name__ == "__main__":
    main()
