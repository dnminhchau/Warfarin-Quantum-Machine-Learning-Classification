"""Mô hình lượng tử: QSVC (quantum kernel SVM) và VQC (variational quantum classifier).

Cả hai mô phỏng chính xác bằng statevector của Qiskit (không nhiễu, không shots). Mạch dùng
thư viện chuẩn của Qiskit (zz_feature_map, real_amplitudes); phép nhân ma trận được tính bằng
numpy để tăng tốc. Hàm self_test() đối chiếu kết quả với Qiskit: python -m src.quantum_models
"""
from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import pauli_feature_map, real_amplitudes, z_feature_map, zz_feature_map
from qiskit.quantum_info import Operator, Statevector
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


# Hàm dùng chung
def make_feature_map(k: int, kind: str = "zz", reps: int = 2, entanglement: str = "linear"):
    """Tạo mạch feature map.

    kind:
        zz        ZZFeatureMap (Havlíček et al., 2019), có rối lượng tử giữa các qubit.
        product   ZFeatureMap, không rối lượng tử; dùng để đối chứng với zz.
        pauli_yzz PauliFeatureMap với cổng Y trên từng qubit và ZZ trên từng cặp qubit.
        angle_ry  Mã hóa góc: mỗi qubit quay RY(x_i) một lần, không rối lượng tử.
    """
    if kind == "zz":
        return zz_feature_map(feature_dimension=k, reps=reps, entanglement=entanglement)
    if kind == "product":
        return z_feature_map(feature_dimension=k, reps=reps)
    if kind == "pauli_yzz":
        return pauli_feature_map(feature_dimension=k, reps=reps, paulis=["Y", "ZZ"], entanglement=entanglement)
    if kind == "angle_ry":
        x = ParameterVector("x", k)
        qc = QuantumCircuit(k)
        for i in range(k):
            qc.ry(x[i], i)
        return qc
    raise ValueError(kind)


def feature_states(X: np.ndarray, fmap) -> np.ndarray:
    """Statevector |φ(x)> của từng mẫu, kích thước (n, 2^k)."""
    return np.array([Statevector(fmap.assign_parameters(x)).data for x in X])


def feature_unitaries(X: np.ndarray, fmap) -> np.ndarray:
    """Ma trận unitary V(x) (2^k x 2^k) của từng mẫu, dùng cho data re-uploading."""
    return np.array([Operator(fmap.assign_parameters(x)).data for x in X])


def z_signs(k: int) -> np.ndarray:
    """Bảng dấu của phép đo Z: phần tử (b, q) = +1 nếu bit q của b bằng 0, ngược lại -1.

    Thứ tự qubit theo quy ước little-endian của Qiskit.
    """
    b = np.arange(2 ** k)[:, None]
    q = np.arange(k)[None, :]
    return 1.0 - 2.0 * ((b >> q) & 1)


def balanced_weights(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Trọng số mẫu theo lớp, cùng công thức class_weight='balanced' của scikit-learn."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return (len(y) / (n_classes * counts))[y]


# QSVC
class QuantumKernelSVC(BaseEstimator, ClassifierMixin):
    """SVM với fidelity quantum kernel K(a, b) = |<φ(a)|φ(b)>|^2, tính bằng statevector.

    Tham số:
        bandwidth  Hệ số nhân vào đầu vào trước khi mã hóa (x -> bandwidth * x). Giá trị nhỏ
                   giúp giảm hiện tượng tập trung của hàm nhân (kernel concentration).
        C          Hệ số phạt của SVM.
    """

    def __init__(self, feature_map: str = "zz", reps: int = 2, entanglement: str = "linear",
                 bandwidth: float = 1.0, C: float = 1.0, class_weight="balanced"):
        self.feature_map = feature_map
        self.reps = reps
        self.entanglement = entanglement
        self.bandwidth = bandwidth
        self.C = C
        self.class_weight = class_weight

    def _states(self, X):
        return feature_states(self.bandwidth * np.asarray(X, float), self.fmap_)

    def fit(self, X, y):
        t0 = time.perf_counter()
        X = np.asarray(X, float)
        self.fmap_ = make_feature_map(X.shape[1], self.feature_map, self.reps, self.entanglement)
        self.train_states_ = self._states(X)
        K = np.abs(self.train_states_.conj() @ self.train_states_.T) ** 2
        off = K[~np.eye(len(K), dtype=bool)]
        # Thống kê phần tử ngoài đường chéo để theo dõi kernel concentration
        self.kernel_offdiag_mean_ = float(off.mean())
        self.kernel_offdiag_std_ = float(off.std())
        self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight=self.class_weight).fit(K, y)
        self.classes_ = self.svc_.classes_
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def predict(self, X):
        S = self._states(X)
        K = np.abs(S.conj() @ self.train_states_.T) ** 2
        return self.svc_.predict(K)


# VQC
class VQCClassifier(BaseEstimator, ClassifierMixin):
    """Variational Quantum Classifier với lớp đọc tuyến tính (hybrid).

    Mạch, lặp lại `layers` lần: |0..0> -> V(x) -> U(θ_1) -> [V(x) -> U(θ_2) ...] -> đo <Z_0>..<Z_{k-1}>.
    Đầu ra: logits = W · <Z> + b, xác suất = softmax(logits), với W kích thước 3 x k.

    Huấn luyện bằng statevector chính xác, gradient theo quy tắc parameter-shift, tối ưu L-BFGS-B.
    Hàm mất mát là cross-entropy có trọng số lớp cộng phạt L2 trên W. Mức đoán ngẫu nhiên
    đều 3 lớp tương ứng loss = ln(3) ≈ 1.0986.
    """

    def __init__(self, reps: int = 2, layers: int = 1, fm_reps: int = 1, entanglement: str = "linear",
                 l2: float = 1e-3, maxiter: int = 300, class_weight="balanced", seed: int = 42,
                 eval_shots: int | None = None, bandwidth: float = 1.0):
        self.reps = reps                  # số lần lặp của ansatz RealAmplitudes
        self.layers = layers              # 1: VQC chuẩn; >1: data re-uploading
        self.fm_reps = fm_reps            # số lần lặp của ZZ feature map
        self.entanglement = entanglement
        self.l2 = l2                      # hệ số phạt L2 trên W
        self.maxiter = maxiter
        self.class_weight = class_weight  # 'balanced' hoặc None
        self.seed = seed                  # seed khởi tạo θ và W
        self.eval_shots = eval_shots      # None: dự đoán chính xác; số nguyên: mô phỏng đo hữu hạn lần
        self.bandwidth = bandwidth        # hệ số nhân đầu vào, như QSVC

    # Dựng mạch
    def _build(self, k: int):
        self.k_ = k
        self.dim_ = 2 ** k
        self.fmap_ = make_feature_map(k, "zz", self.fm_reps, self.entanglement)
        self.ansatz_ = real_amplitudes(num_qubits=k, reps=self.reps, entanglement=self.entanglement)
        self.p_block_ = self.ansatz_.num_parameters
        self.n_theta_ = self.layers * self.p_block_
        self.zs_ = z_signs(k)
        self._compile_ansatz()

    def _encode(self, X):
        """Mã hóa dữ liệu: layers=1 dùng statevector |φ(x)>, layers>1 dùng unitary V(x)."""
        X = self.bandwidth * np.asarray(X, float)
        if self.layers == 1:
            return feature_states(X, self.fmap_)
        return feature_unitaries(X, self.fmap_)

    def _compile_ansatz(self):
        """Đọc các cổng của ansatz một lần để ghép ma trận bằng numpy.

        Cổng CX lấy ma trận từ Qiskit; cổng RY(θ) tính theo công thức 2x2.
        Tính đúng đắn được kiểm tra trong self_test().
        """
        k = self.k_
        index = {p: i for i, p in enumerate(self.ansatz_.parameters)}
        self.ops_ = []
        for ins in self.ansatz_.data:
            qs = [self.ansatz_.find_bit(q).index for q in ins.qubits]
            if ins.operation.name == "ry":
                self.ops_.append(("ry", qs[0], index[ins.operation.params[0]]))
            elif ins.operation.name == "cx":
                sub = QuantumCircuit(k)
                sub.cx(qs[0], qs[1])
                self.ops_.append(("fixed", Operator(sub).data))
            else:
                raise ValueError(f"Cổng chưa hỗ trợ trong ansatz: {ins.operation.name}")

    def _ansatz_unitary(self, theta_block):
        """Ma trận 2^k x 2^k của một khối ansatz với tham số θ."""
        k = self.k_
        U = np.eye(self.dim_, dtype=complex)
        for op in self.ops_:
            if op[0] == "fixed":
                G = op[1]
            else:
                _, q, j = op
                c, s_ = np.cos(theta_block[j] / 2), np.sin(theta_block[j] / 2)
                ry = np.array([[c, -s_], [s_, c]])
                # Little-endian: qubit q ở vị trí thứ q tính từ phải trong tích Kronecker
                G = np.kron(np.kron(np.eye(2 ** (k - 1 - q)), ry), np.eye(2 ** q))
            U = G @ U
        return U

    def _expect_z(self, enc, Us):
        """Kỳ vọng <Z_q> của mọi mẫu, kích thước (n, k)."""
        if self.layers == 1:
            psi = enc @ Us[0].T
        else:
            psi = np.zeros((enc.shape[0], self.dim_), complex)
            psi[:, 0] = 1.0
            for U in Us:
                psi = np.einsum("nij,nj->ni", enc, psi)
                psi = psi @ U.T
        probs = np.abs(psi) ** 2
        return probs @ self.zs_

    def _split(self, params):
        """Tách vector tham số thành (θ, W, b)."""
        t = params[: self.n_theta_]
        W = params[self.n_theta_: self.n_theta_ + 3 * self.k_].reshape(3, self.k_)
        b = params[self.n_theta_ + 3 * self.k_:]
        return t, W, b

    def _blocks(self, theta):
        return [self._ansatz_unitary(theta[l * self.p_block_:(l + 1) * self.p_block_]) for l in range(self.layers)]

    # Hàm mất mát và gradient
    def _loss_grad(self, params, enc, Y, w):
        theta, W, b = self._split(params)
        Us = self._blocks(theta)
        E = self._expect_z(enc, Us)
        logits = E @ W.T + b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        sw = w / w.sum()
        loss = -np.sum(sw * np.log(np.clip((P * Y).sum(1), 1e-12, None))) + self.l2 * np.sum(W ** 2)
        self._last_loss = float(loss)

        dlogits = (P - Y) * sw[:, None]
        gW = dlogits.T @ E + 2 * self.l2 * W
        gb = dlogits.sum(0)
        dE = dlogits @ W

        # Parameter-shift: d<Z>/dθ = [<Z>(θ + π/2) - <Z>(θ - π/2)] / 2
        gt = np.zeros_like(theta)
        for j in range(self.n_theta_):
            l = j // self.p_block_
            plus, minus = theta.copy(), theta.copy()
            plus[j] += np.pi / 2
            minus[j] -= np.pi / 2
            Up, Um = list(Us), list(Us)
            Up[l] = self._ansatz_unitary(plus[l * self.p_block_:(l + 1) * self.p_block_])
            Um[l] = self._ansatz_unitary(minus[l * self.p_block_:(l + 1) * self.p_block_])
            dEdj = (self._expect_z(enc, Up) - self._expect_z(enc, Um)) / 2
            gt[j] = np.sum(dE * dEdj)
        return loss, np.concatenate([gt, gW.ravel(), gb])

    # API theo chuẩn scikit-learn
    def fit(self, X, y):
        t0 = time.perf_counter()
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        self.classes_ = np.arange(3)
        self._build(X.shape[1])
        enc = self._encode(X)
        Y = np.eye(3)[y]
        w = balanced_weights(y) if self.class_weight == "balanced" else np.ones(len(y))
        rng = np.random.default_rng(self.seed)
        x0 = np.concatenate([rng.uniform(-np.pi, np.pi, self.n_theta_),
                             rng.normal(0, 0.1, 3 * self.k_),
                             np.zeros(3)])
        self.loss_history_ = []
        res = minimize(self._loss_grad, x0, args=(enc, Y, w), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.maxiter},
                       callback=lambda p: self.loss_history_.append(self._last_loss))
        self.params_ = res.x
        self.initial_loss_ = float(self._loss_grad(x0, enc, Y, w)[0])
        self.final_loss_ = float(res.fun)
        self.n_iter_ = int(res.nit)
        self.converged_ = bool(res.success)
        self.uniform_loss_ = float(np.log(3))
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def predict_proba(self, X):
        theta, W, b = self._split(self.params_)
        E = self._expect_z(self._encode(np.asarray(X, float)), self._blocks(theta))
        if self.eval_shots:
            rng = np.random.default_rng(self.seed + 1)
            p1 = (1 - E) / 2
            E = 1 - 2 * rng.binomial(self.eval_shots, np.clip(p1, 0, 1)) / self.eval_shots
        logits = E @ W.T + b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def circuit_stats(self) -> dict:
        """Số qubit, độ sâu và số cổng CNOT của mạch (trước transpile)."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.k_)
        for _ in range(self.layers):
            qc.compose(self.fmap_, inplace=True)
            qc.compose(self.ansatz_, inplace=True)
        d = qc.decompose(reps=3)
        return {"qubits": self.k_, "depth": d.depth(), "cx": int(d.count_ops().get("cx", 0)),
                "trainable_quantum_params": self.n_theta_, "classical_readout_params": 3 * self.k_ + 3}


# Kiểm tra đối chiếu với Qiskit
def self_test(seed: int = 0) -> None:
    """Đối chiếu kernel, VQC forward và gradient với Qiskit hoặc sai phân hữu hạn."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, np.pi, (12, 4))
    y = np.array([0, 1, 2] * 4)

    # (1) Kernel so với FidelityStatevectorKernel của qiskit-machine-learning
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    m = QuantumKernelSVC(bandwidth=0.7).fit(X, y)
    ref = FidelityStatevectorKernel(feature_map=make_feature_map(4)).evaluate(0.7 * X)
    mine = np.abs(m.train_states_.conj() @ m.train_states_.T) ** 2
    print("Kernel khớp Qiskit:", "OK" if np.allclose(ref, mine, atol=1e-10) else "SAI")

    # (2) VQC forward so với Statevector của mạch đầy đủ, với layers=1 và layers=2
    for layers in (1, 2):
        v = VQCClassifier(reps=2, layers=layers, maxiter=0)
        v._build(4)
        theta = rng.uniform(-np.pi, np.pi, v.n_theta_)
        E_fast = v._expect_z(v._encode(X[:3]), v._blocks(theta))
        E_ref = []
        for x in X[:3]:
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(4)
            for l in range(layers):
                qc.compose(v.fmap_.assign_parameters(x), inplace=True)
                qc.compose(v.ansatz_.assign_parameters(theta[l * v.p_block_:(l + 1) * v.p_block_]), inplace=True)
            p = Statevector(qc).probabilities()
            E_ref.append(p @ z_signs(4))
        print(f"VQC forward (layers={layers}) khớp Qiskit:", "OK" if np.allclose(E_fast, E_ref, atol=1e-10) else "SAI")

    # (3) Gradient parameter-shift so với sai phân hữu hạn
    v = VQCClassifier(reps=1, layers=2, maxiter=0)
    v._build(4)
    enc = v._encode(X)
    Y = np.eye(3)[y]
    w = balanced_weights(y)
    p0 = rng.normal(0, 1, v.n_theta_ + 3 * 4 + 3)
    _, g = v._loss_grad(p0, enc, Y, w)
    eps = 1e-6
    g_fd = np.array([(v._loss_grad(p0 + eps * e, enc, Y, w)[0] - v._loss_grad(p0 - eps * e, enc, Y, w)[0]) / (2 * eps)
                     for e in np.eye(len(p0))])
    print("Gradient parameter-shift khớp sai phân:", "OK" if np.allclose(g, g_fd, atol=1e-5) else "SAI")


if __name__ == "__main__":
    self_test()
