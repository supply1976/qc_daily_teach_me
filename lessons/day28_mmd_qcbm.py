"""Day 28: 用 Maximum Mean Discrepancy 從樣本訓練 QCBM。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


NUMBER_OF_QUBITS = 3
DIMENSION = 2**NUMBER_OF_QUBITS


def qcbm_state(parameters: np.ndarray) -> Statevector:
    """建立可表示三 bit even-parity 分布的 Born machine。"""
    circuit = QuantumCircuit(NUMBER_OF_QUBITS)

    for qubit, angle in enumerate(parameters):
        circuit.ry(float(angle), qubit)

    # q2 <- q2 XOR q0 XOR q1；當 q2=0 時只會生成偶同位字串。
    circuit.cx(0, 2)
    circuit.cx(1, 2)

    return Statevector.from_instruction(circuit)


def hamming_kernel(bandwidth: float = 1.0) -> np.ndarray:
    """K[x,y] = exp(-Hamming(x,y) / bandwidth)。"""
    integers = np.arange(DIMENSION, dtype=np.uint8)
    bit_positions = np.arange(NUMBER_OF_QUBITS, dtype=np.uint8)
    bits = (integers[:, None] >> bit_positions) & 1
    distances = np.sum(
        bits[:, None, :] != bits[None, :, :],
        axis=2,
    )
    return np.exp(-distances / bandwidth)


def counts_to_distribution(counts: dict[str, int]) -> np.ndarray:
    """把 Qiskit bitstring counts 轉成 [000,...,111] 機率向量。"""
    shots = sum(counts.values())
    distribution = np.zeros(DIMENSION)

    for bits, count in counts.items():
        distribution[int(bits, 2)] = count / shots

    return distribution


def sample_model(
    parameters: np.ndarray,
    shots: int,
    seed: int,
) -> np.ndarray:
    state = qcbm_state(parameters)
    state.seed(seed)
    counts = dict(state.sample_counts(shots=shots))
    return counts_to_distribution(counts)


def mmd_squared(
    first: np.ndarray,
    second: np.ndarray,
    kernel: np.ndarray,
) -> float:
    """Biased empirical MMD^2；以 empirical distributions 計算。"""
    difference = first - second
    return float(difference @ kernel @ difference)


def total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(first - second)))


def sampled_mmd_gradient(
    parameters: np.ndarray,
    target_samples: np.ndarray,
    kernel: np.ndarray,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """以 current/shifted circuits 的樣本估計 parameter-shift gradient。"""
    model_samples = sample_model(parameters, shots, seed)
    difference = model_samples - target_samples
    gradient = np.zeros_like(parameters)

    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += np.pi / 2
        minus[index] -= np.pi / 2

        shifted_plus = sample_model(
            plus,
            shots,
            seed + 10 + 2 * index,
        )
        shifted_minus = sample_model(
            minus,
            shots,
            seed + 11 + 2 * index,
        )

        # d(MMD^2)/d theta = (p_plus-p_minus)^T K (p-q)
        gradient[index] = (
            (shifted_plus - shifted_minus)
            @ kernel
            @ difference
        )

    sampled_loss = mmd_squared(
        model_samples,
        target_samples,
        kernel,
    )
    return gradient, sampled_loss


def train_with_sampled_mmd(
    target_samples: np.ndarray,
    kernel: np.ndarray,
    shots: int = 4000,
    steps: int = 200,
) -> np.ndarray:
    """用 sampled parameter-shift gradient 與 Adam 最小化 MMD。"""
    parameters = np.array([0.35, -0.55, 0.25])
    learning_rate = 0.08
    beta_1 = 0.9
    beta_2 = 0.999
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)

    print(" step   sampled MMD^2   exact MMD^2      TV distance")
    print("-------------------------------------------------------")

    checkpoints = {0, 1, 10, 25, 50, 100, 150, 200}

    def report(step: int) -> None:
        exact_model = qcbm_state(parameters).probabilities()
        sampled_model = sample_model(
            parameters,
            shots,
            38000 + step,
        )
        print(
            f"{step:5d}   "
            f"{mmd_squared(sampled_model, target_samples, kernel):.9f}   "
            f"{mmd_squared(exact_model, TARGET, kernel):.9f}   "
            f"{total_variation(exact_model, TARGET):.9f}"
        )

    report(0)

    for step in range(1, steps + 1):
        gradient, _ = sampled_mmd_gradient(
            parameters,
            target_samples,
            kernel,
            shots,
            seed=28000 + 100 * step,
        )

        first_moment = (
            beta_1 * first_moment
            + (1.0 - beta_1) * gradient
        )
        second_moment = (
            beta_2 * second_moment
            + (1.0 - beta_2) * gradient**2
        )
        corrected_first = first_moment / (1.0 - beta_1**step)
        corrected_second = second_moment / (1.0 - beta_2**step)

        parameters -= (
            learning_rate
            * corrected_first
            / (np.sqrt(corrected_second) + 1e-8)
        )

        if step in checkpoints:
            report(step)

    return parameters


# Qiskit probabilities 使用 |q2 q1 q0> 順序。
# 偶同位資料：000、011、101、110，各占 1/4。
TARGET = np.array([
    0.25,
    0.0,
    0.0,
    0.25,
    0.0,
    0.25,
    0.25,
    0.0,
])


def main() -> None:
    kernel = hamming_kernel(bandwidth=1.0)

    # 固定一份 classical dataset；訓練只存取其 samples/histogram。
    data = np.repeat([0, 3, 5, 6], repeats=1000)
    target_samples = np.bincount(
        data,
        minlength=DIMENSION,
    ) / data.size

    print("Target distribution [000, ..., 111]:")
    print(TARGET)
    print("Empirical training data:")
    print(target_samples)
    print()

    parameters = train_with_sampled_mmd(
        target_samples,
        kernel,
    )
    learned = qcbm_state(parameters).probabilities()

    final_state = qcbm_state(parameters)
    final_state.seed(2899)
    counts = {
        str(bits): int(count)
        for bits, count
        in final_state.sample_counts(shots=10000).items()
    }

    odd_parity_probability = float(
        learned[[1, 2, 4, 7]].sum()
    )

    print("\nLearned exact distribution:")
    print(np.round(learned, 8))
    print("Learned parameters:", np.round(parameters, 6))
    print(
        "Exact MMD^2:       ",
        f"{mmd_squared(learned, TARGET, kernel):.9e}",
    )
    print(
        "TV distance:       ",
        f"{total_variation(learned, TARGET):.9e}",
    )
    print(
        "Odd-parity mass:   ",
        f"{odd_parity_probability:.9e}",
    )
    print("Generated counts:   ", counts)


if __name__ == "__main__":
    main()
