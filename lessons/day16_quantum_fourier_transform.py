"""第 16 課：量子傅立葉轉換、相位梯度與 inverse QFT。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


def qft_circuit(number_of_qubits: int) -> QuantumCircuit:
    """以 H、controlled-phase 與 SWAP 建立 QFT 電路。"""
    if number_of_qubits < 1:
        raise ValueError("number_of_qubits must be positive")

    circuit = QuantumCircuit(number_of_qubits, name="QFT")

    # 從最高編號 qubit 開始，逐步建立不同尺度的相位。
    for target in reversed(range(number_of_qubits)):
        circuit.h(target)

        for control in reversed(range(target)):
            distance = target - control
            angle = np.pi / (2**distance)
            circuit.cp(angle, control, target)

    # 上述分解會產生反轉的 qubit 順序，因此最後交換回來。
    for qubit in range(number_of_qubits // 2):
        circuit.swap(qubit, number_of_qubits - qubit - 1)

    return circuit


def exact_qft_amplitudes(value: int, dimension: int) -> np.ndarray:
    """直接由 QFT 矩陣公式計算 QFT|value>。"""
    output_indices = np.arange(dimension)
    return np.exp(
        2j * np.pi * value * output_indices / dimension
    ) / np.sqrt(dimension)


def main() -> None:
    number_of_qubits = 3
    dimension = 2**number_of_qubits
    input_value = 5  # |101>

    circuit = qft_circuit(number_of_qubits)
    input_state = Statevector.from_int(input_value, dims=dimension)
    qft_state = input_state.evolve(circuit)

    # 與 N x N 傅立葉矩陣的解析公式交叉驗證。
    expected = exact_qft_amplitudes(input_value, dimension)
    maximum_error = float(np.max(np.abs(qft_state.data - expected)))

    probabilities = qft_state.probabilities()
    phases_degrees = np.rad2deg(np.angle(qft_state.data))

    # QFT 的 inverse 應精確恢復原 computational-basis state。
    recovered_state = qft_state.evolve(circuit.inverse())
    fidelity = float(state_fidelity(input_state, recovered_state))

    recovered_state.seed(1600)
    counts = {
        str(outcome): int(count)
        for outcome, count in recovered_state.sample_counts(
            shots=1024
        ).items()
    }

    print("Three-qubit QFT circuit:")
    print(circuit.draw(output="text"))
    print(f"\nInput state: |{input_value:03b}>")

    print("\nQFT amplitudes:")
    print(np.round(qft_state.data, 6))

    print("\nComputational-basis probabilities:")
    print(np.round(probabilities, 6))

    print("\nPhases in degrees:")
    print(np.round(phases_degrees, 3))

    print(f"\nMaximum error versus QFT formula: {maximum_error:.3e}")
    print(f"Inverse-QFT recovery fidelity:     {fidelity:.12f}")
    print(f"Recovered counts:                 {counts}")


if __name__ == "__main__":
    main()
