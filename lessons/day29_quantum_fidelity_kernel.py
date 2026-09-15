"""Day 29: 量子 fidelity kernel、Hamming distance 與 SWAP test。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


NUMBER_OF_QUBITS = 3
DIMENSION = 2**NUMBER_OF_QUBITS
BITSTRINGS = [
    format(value, f"0{NUMBER_OF_QUBITS}b")
    for value in range(DIMENSION)
]


def feature_circuit(bitstring: str, angle: float) -> QuantumCircuit:
    """將 binary string x 編碼成 tensor-product quantum feature state。"""
    circuit = QuantumCircuit(NUMBER_OF_QUBITS)

    # Qiskit 顯示 basis states 時採 |q2 q1 q0> 順序。
    for qubit, bit in enumerate(reversed(bitstring)):
        circuit.ry(angle * int(bit), qubit)

    return circuit


def feature_state(bitstring: str, angle: float) -> Statevector:
    return Statevector.from_instruction(
        feature_circuit(bitstring, angle)
    )


def fidelity_kernel(
    first: str,
    second: str,
    angle: float,
) -> float:
    """k(x,y) = |<phi(x)|phi(y)>|^2。"""
    first_state = feature_state(first, angle)
    second_state = feature_state(second, angle)
    overlap = np.vdot(first_state.data, second_state.data)
    return float(abs(overlap) ** 2)


def quantum_kernel_matrix(
    bitstrings: list[str],
    angle: float,
) -> np.ndarray:
    states = [feature_state(bits, angle) for bits in bitstrings]
    size = len(states)
    kernel = np.zeros((size, size))

    for row in range(size):
        for column in range(size):
            overlap = np.vdot(
                states[row].data,
                states[column].data,
            )
            kernel[row, column] = abs(overlap) ** 2

    return kernel


def hamming_kernel_matrix(
    bitstrings: list[str],
    bandwidth: float,
) -> np.ndarray:
    size = len(bitstrings)
    kernel = np.zeros((size, size))

    for row, first in enumerate(bitstrings):
        for column, second in enumerate(bitstrings):
            distance = sum(
                left != right
                for left, right in zip(first, second)
            )
            kernel[row, column] = np.exp(
                -distance / bandwidth
            )

    return kernel


def swap_test_circuit(
    first: str,
    second: str,
    angle: float,
) -> QuantumCircuit:
    """用一個 ancilla 比較兩個三 qubit feature states。"""
    total_qubits = 1 + 2 * NUMBER_OF_QUBITS
    circuit = QuantumCircuit(total_qubits)

    first_register = list(range(1, 1 + NUMBER_OF_QUBITS))
    second_register = list(
        range(1 + NUMBER_OF_QUBITS, total_qubits)
    )

    circuit.compose(
        feature_circuit(first, angle),
        qubits=first_register,
        inplace=True,
    )
    circuit.compose(
        feature_circuit(second, angle),
        qubits=second_register,
        inplace=True,
    )

    circuit.h(0)
    for left, right in zip(first_register, second_register):
        circuit.cswap(0, left, right)
    circuit.h(0)

    return circuit


def sampled_swap_kernel(
    first: str,
    second: str,
    angle: float,
    shots: int,
    seed: int,
) -> tuple[float, dict[str, int]]:
    """由 ancilla samples 估計 fidelity：F = 2 P(0) - 1。"""
    state = Statevector.from_instruction(
        swap_test_circuit(first, second, angle)
    )
    state.seed(seed)
    raw_counts = state.sample_counts(shots=shots, qargs=[0])
    counts = {
        str(outcome): int(count)
        for outcome, count in raw_counts.items()
    }
    probability_zero = counts.get("0", 0) / shots
    estimate = 2.0 * probability_zero - 1.0

    return estimate, counts


def mmd_squared(
    first: np.ndarray,
    second: np.ndarray,
    kernel: np.ndarray,
) -> float:
    difference = first - second
    return float(difference @ kernel @ difference)


def main() -> None:
    bandwidth = 1.0

    # 讓 cos^2(angle/2) = exp(-1 / bandwidth)。
    angle = 2.0 * np.arccos(
        np.exp(-1.0 / (2.0 * bandwidth))
    )

    quantum_kernel = quantum_kernel_matrix(BITSTRINGS, angle)
    classical_kernel = hamming_kernel_matrix(
        BITSTRINGS,
        bandwidth,
    )

    maximum_error = float(
        np.max(np.abs(quantum_kernel - classical_kernel))
    )
    eigenvalues = np.linalg.eigvalsh(quantum_kernel)

    even_parity = np.zeros(DIMENSION)
    even_parity[[0, 3, 5, 6]] = 0.25
    odd_parity = np.zeros(DIMENSION)
    odd_parity[[1, 2, 4, 7]] = 0.25

    print(f"Bandwidth sigma:       {bandwidth:.3f}")
    print(f"Encoding angle theta:  {angle:.9f} rad")
    print(
        "cos^2(theta / 2):   "
        f"{np.cos(angle / 2.0) ** 2:.9f}"
    )
    print("exp(-1 / sigma):       " f"{np.exp(-1 / bandwidth):.9f}")

    print("\nQuantum fidelity kernel:")
    print(np.round(quantum_kernel, 4))
    print(
        "\nMaximum error versus Hamming kernel: "
        f"{maximum_error:.3e}"
    )
    print("Kernel eigenvalues:")
    print(np.round(eigenvalues, 9))
    print(
        "MMD^2(even parity, odd parity): "
        f"{mmd_squared(even_parity, odd_parity, quantum_kernel):.9f}"
    )

    shots = 8192
    print(f"\nSWAP-test estimates ({shots} shots):")

    for index, (first, second) in enumerate([
        ("000", "001"),
        ("000", "111"),
    ]):
        exact = fidelity_kernel(first, second, angle)
        estimate, counts = sampled_swap_kernel(
            first,
            second,
            angle,
            shots,
            seed=2900 + index,
        )
        distance = sum(
            left != right
            for left, right in zip(first, second)
        )

        print(
            f"{first} vs {second}: "
            f"d_H={distance}, "
            f"exact={exact:.9f}, "
            f"sampled={estimate:.9f}, "
            f"counts={counts}"
        )


if __name__ == "__main__":
    main()
