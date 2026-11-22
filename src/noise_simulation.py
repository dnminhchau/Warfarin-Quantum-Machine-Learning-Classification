"""Mô phỏng mạch lượng tử có nhiễu bằng ma trận mật độ (density matrix), tính theo lô bằng numpy.

Mô hình nhiễu theo cách mô tả phổ biến của máy lượng tử siêu dẫn (ví dụ IBM):
- Mạch được transpile về các cổng gốc cx, rz, sx, x trên các qubit nối thành một hàng (coupling map dạng line).
- Sau mỗi cổng sx, x (và sxdg trong mạch nghịch đảo): nhiễu khử cực (depolarizing) 1 qubit, tham số p1.
- Sau mỗi cổng cx: nhiễu khử cực 2 qubit, tham số p2.
- Cổng rz được thực hiện bằng phần mềm (virtual Z) nên không có lỗi.
- Lỗi đọc (readout error): mỗi qubit bị đọc nhầm 0 <-> 1 với xác suất r.
Nhiễu khử cực tham số λ trên tập qubit S: ρ -> (1 - λ) ρ + λ (I/2^m ⊗ Tr_S ρ), cùng định nghĩa với
depolarizing_error của Qiskit Aer.

Hàm nhân của QSVC trên phần cứng được ước lượng bằng mạch U(x_i) rồi U(x_j)†, đo xác suất ra toàn 0.
Thay vì mô phỏng riêng từng cặp, ta tính ρ(x_i) (mạch U có nhiễu) và M(x_j) (tác động ngược của mạch U(x_j)†
có nhiễu lên phép đo toàn 0, theo bức tranh Heisenberg). Khi đó K(i, j) = Tr(M(x_j) ρ(x_i)), nên cả ma trận
hàm nhân là một phép nhân ma trận. Kết quả trùng với mô phỏng từng mạch bằng Qiskit Aer (xem self_test()).
Chạy kiểm tra: python -m src.noise_simulation (cần qiskit-aer)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import Operator, Pauli
from qiskit.transpiler import CouplingMap

BASIS = ["cx", "rz", "sx", "x"]
NOISY_1Q = {"sx", "x"}


@dataclass(frozen=True)
class NoiseModel:
    p1: float        # tham số khử cực sau mỗi cổng 1 qubit (sx, x)
    p2: float        # tham số khử cực sau mỗi cổng cx
    readout: float   # xác suất đọc nhầm mỗi qubit

    def as_dict(self):
        return asdict(self)


def transpile_line(circ: QuantumCircuit, seed: int = 0) -> QuantumCircuit:
    """Transpile về cổng gốc trên k qubit nối thành hàng, giữ nguyên thứ tự qubit."""
    k = circ.num_qubits
    tc = transpile(circ, basis_gates=BASIS, coupling_map=CouplingMap.from_line(k),
                   initial_layout=list(range(k)), optimization_level=1, seed_transpiler=seed)
    if tc.layout is not None and tc.layout.final_index_layout() != list(range(k)):
        raise RuntimeError("Transpile làm đổi thứ tự qubit; cần xử lý thêm.")
    return tc


def gate_counts(tc: QuantumCircuit) -> dict:
    ops = tc.count_ops()
    return {"depth": tc.depth(), "cx": int(ops.get("cx", 0)),
            "one_qubit_noisy": int(sum(ops.get(g, 0) for g in NOISY_1Q)), "rz": int(ops.get("rz", 0))}


@lru_cache(maxsize=None)
def _gate_matrix(name: str, qubits: tuple, k: int, param: float | None = None) -> np.ndarray:
    qc = QuantumCircuit(k)
    getattr(qc, name)(*(([param] if param is not None else []) + list(qubits)))
    return Operator(qc).data


@lru_cache(maxsize=None)
def _paulis(qubits: tuple, k: int) -> np.ndarray:
    """Mọi tích Pauli trên các qubit cho trước (kể cả I), dạng ma trận 2^k x 2^k."""
    mats = []
    for combo in product("IXYZ", repeat=len(qubits)):
        label = ["I"] * k
        for q, c in zip(qubits, combo):
            label[k - 1 - q] = c        # nhãn Pauli của Qiskit: ký tự bên phải là qubit 0
        mats.append(Pauli("".join(label)).to_matrix())
    return np.array(mats)


def _depolarize(rho: np.ndarray, qubits: tuple, lam: float, k: int) -> np.ndarray:
    if lam <= 0:
        return rho
    P = _paulis(qubits, k)
    twirl = sum(p @ rho @ p for p in P) / len(P)   # = I/2^m ⊗ Tr_S ρ
    return (1 - lam) * rho + lam * twirl


class Program:
    """Danh sách cổng của mạch đã transpile; góc rz phụ thuộc dữ liệu được tính riêng cho từng mẫu."""

    def __init__(self, tc: QuantumCircuit):
        self.tc = tc
        self.k = tc.num_qubits
        self.ops = []          # ("rz", qubit, cột góc) hoặc ("gate", tên, qubits, ma trận)
        self.param_index = []  # vị trí lệnh rz có tham số trong tc.data
        for j, ins in enumerate(tc.data):
            name = ins.operation.name
            qs = tuple(tc.find_bit(q).index for q in ins.qubits)
            if name == "barrier":
                continue
            if name == "rz" and isinstance(ins.operation.params[0], ParameterExpression):
                self.ops.append(("rz", qs[0], len(self.param_index)))
                self.param_index.append(j)
            elif name in BASIS:
                p = float(ins.operation.params[0]) if ins.operation.params else None
                self.ops.append(("gate", name, qs, _gate_matrix(name, qs, self.k, p)))
            else:
                raise ValueError(f"Cổng chưa hỗ trợ: {name}")
        b = np.arange(2 ** self.k)[:, None]
        self.bits = ((b >> np.arange(self.k)[None, :]) & 1).astype(float)   # (2^k, k), little-endian

    def angles(self, X: np.ndarray) -> np.ndarray:
        """Góc của các cổng rz có tham số, kích thước (n, số cổng)."""
        params = list(self.tc.parameters)
        out = np.empty((len(X), len(self.param_index)))
        for i, x in enumerate(np.asarray(X, float)):
            bound = self.tc.assign_parameters(dict(zip(params, x)))
            out[i] = [float(bound.data[j].operation.params[0]) for j in self.param_index]
        return out

    def _noise(self, name, qs, noise):
        if noise is None:
            return 0.0
        return noise.p2 if name == "cx" else (noise.p1 if name in NOISY_1Q else 0.0)

    def forward(self, X: np.ndarray, noise: NoiseModel | None) -> np.ndarray:
        """ρ(x) = mạch U(x) có nhiễu tác động lên |0..0>, kích thước (n, 2^k, 2^k)."""
        A = self.angles(X)
        d = 2 ** self.k
        rho = np.zeros((len(A), d, d), complex)
        rho[:, 0, 0] = 1.0
        for op in self.ops:
            if op[0] == "rz":
                _, q, c = op
                z = 1 - 2 * self.bits[:, q]                       # +1 nếu bit q = 0
                ph = np.exp(-0.5j * A[:, c:c + 1] * z[None, :])  # rz(θ) = diag(e^{-iθ/2}, e^{iθ/2})
                rho = rho * ph[:, :, None] * ph.conj()[:, None, :]
            else:
                _, name, qs, G = op
                rho = G @ rho @ G.conj().T
                rho = _depolarize(rho, qs, self._noise(name, qs, noise), self.k)
        return rho

    def zero_effect(self, noise: NoiseModel | None) -> np.ndarray:
        """Phép đo 'đọc ra toàn 0' khi có lỗi đọc: ma trận chéo Π_q P(đọc 0 | bit q)."""
        r = 0.0 if noise is None else noise.readout
        return np.diag(np.prod(np.where(self.bits == 0, 1 - r, r), axis=1)).astype(complex)

    def backward(self, X: np.ndarray, noise: NoiseModel | None) -> np.ndarray:
        """M(x) = tác động ngược của mạch U(x)† có nhiễu lên phép đo toàn 0, kích thước (n, 2^k, 2^k).

        Mạch U(x)† gồm các cổng nghịch đảo theo thứ tự ngược, mỗi cổng có nhiễu cùng loại. Trong bức tranh
        Heisenberg, duyệt các cổng h của U(x) theo thứ tự xuôi: O <- N(O) rồi O <- h O h†
        (kênh khử cực là tự liên hợp).
        """
        A = self.angles(X)
        O = np.repeat(self.zero_effect(noise)[None], len(A), axis=0)
        for op in self.ops:
            if op[0] == "rz":
                _, q, c = op
                z = 1 - 2 * self.bits[:, q]
                ph = np.exp(-0.5j * A[:, c:c + 1] * z[None, :])
                O = O * ph[:, :, None] * ph.conj()[:, None, :]
            else:
                _, name, qs, G = op
                O = _depolarize(O, qs, self._noise(name, qs, noise), self.k)
                O = G @ O @ G.conj().T
        return O

    def prob_one(self, X: np.ndarray, noise: NoiseModel | None) -> np.ndarray:
        """Xác suất đọc ra 1 trên từng qubit sau mạch U(x) có nhiễu và lỗi đọc, kích thước (n, k)."""
        rho = self.forward(X, noise)
        p = np.real(np.einsum("nii->ni", rho)) @ self.bits
        r = 0.0 if noise is None else noise.readout
        return r + (1 - 2 * r) * p


def kernel_from(rho: np.ndarray, M: np.ndarray) -> np.ndarray:
    """K(i, j) = Tr(M_j ρ_i): xác suất đọc ra toàn 0 của mạch U(x_i) rồi U(x_j)†."""
    n, m = len(rho), len(M)
    return np.real(rho.reshape(n, -1) @ M.reshape(m, -1).conj().T)


def sample_shots(p: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Ước lượng xác suất bằng tần suất trong `shots` lần đo."""
    return rng.binomial(shots, np.clip(p, 0, 1)) / shots


def project_psd(K: np.ndarray) -> np.ndarray:
    """Chiếu ma trận đối xứng về nửa xác định dương bằng cách bỏ trị riêng âm (như enforce_psd của Qiskit)."""
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.clip(w, 0, None)) @ V.T


def self_test(seed: int = 0) -> None:
    """Đối chiếu hàm nhân và xác suất đo với Qiskit Aer (density matrix) trên vài mẫu ngẫu nhiên."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel as AerNoise, ReadoutError, depolarizing_error
    from src.quantum_models import make_feature_map

    noise = NoiseModel(p1=1e-3, p2=1e-2, readout=3e-2)
    aer_noise = AerNoise(basis_gates=BASIS + ["sxdg"])
    aer_noise.add_all_qubit_quantum_error(depolarizing_error(noise.p1, 1), ["sx", "x", "sxdg"])
    aer_noise.add_all_qubit_quantum_error(depolarizing_error(noise.p2, 2), ["cx"])
    sim = AerSimulator(method="density_matrix", noise_model=aer_noise)
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, np.pi, (4, 4))
    r = noise.readout

    def p_zero_with_readout(probs):
        bits = np.arange(len(probs))[:, None] >> np.arange(4)[None, :] & 1
        return float(probs @ np.prod(np.where(bits == 0, 1 - r, r), axis=1))

    for kind in ("zz", "product"):
        tc = transpile_line(make_feature_map(4, kind, 2, "linear"))
        prog = Program(tc)
        K = kernel_from(prog.forward(X, noise), prog.backward(X, noise))
        ref = np.zeros_like(K)
        params = list(tc.parameters)
        for i in range(len(X)):
            for j in range(len(X)):
                qc = tc.assign_parameters(dict(zip(params, X[i])))
                qc = qc.compose(tc.assign_parameters(dict(zip(params, X[j]))).inverse())
                qc.save_density_matrix()
                dm = sim.run(qc).result().data()["density_matrix"]
                ref[i, j] = p_zero_with_readout(np.real(np.diag(np.asarray(dm))))
        print(f"Hàm nhân có nhiễu ({kind}) khớp Aer:", "OK" if np.allclose(K, ref, atol=1e-10) else "SAI",
              f"(sai khác lớn nhất {np.abs(K - ref).max():.1e})")

    # Xác suất đọc ra 1 trên từng qubit của một mạch kiểu VQC (feature map + RealAmplitudes)
    from qiskit.circuit.library import real_amplitudes
    ans = real_amplitudes(4, reps=2, entanglement="linear")
    circ = make_feature_map(4, "zz", 1, "linear").compose(ans.assign_parameters(rng.uniform(-np.pi, np.pi, ans.num_parameters)))
    tc = transpile_line(circ)
    prog = Program(tc)
    P1 = prog.prob_one(X, noise)
    ref = []
    for x in X:
        qc = tc.assign_parameters(dict(zip(list(tc.parameters), x)))
        qc.save_density_matrix()
        probs = np.real(np.diag(np.asarray(sim.run(qc).result().data()["density_matrix"])))
        ref.append(r + (1 - 2 * r) * (probs @ prog.bits))
    print("Xác suất đo từng qubit (mạch VQC) khớp Aer:", "OK" if np.allclose(P1, ref, atol=1e-10) else "SAI")

    # Lỗi đọc: so với Aer chạy có shots (kiểm tra thống kê, 400.000 shots)
    aer_noise.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    sim_ro = AerSimulator(method="density_matrix", noise_model=aer_noise, seed_simulator=seed)
    qc = tc.assign_parameters(dict(zip(list(tc.parameters), X[0])))
    qc.measure_all()
    counts = sim_ro.run(qc, shots=400_000).result().get_counts()
    emp = np.zeros(4)
    for bitstring, c in counts.items():
        emp += c * np.array([int(bitstring[::-1][q]) for q in range(4)])
    emp /= 400_000
    z = np.abs(emp - P1[0]) / np.sqrt(P1[0] * (1 - P1[0]) / 400_000)
    print("Lỗi đọc khớp Aer (shots):", "OK" if (z < 4).all() else "SAI", f"(|z| lớn nhất {z.max():.2f})")


if __name__ == "__main__":
    self_test()
