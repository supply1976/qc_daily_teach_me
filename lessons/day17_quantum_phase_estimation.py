"""第 17 課：量子相位估計、controlled powers 與 inverse QFT。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def qft_circuit(number_of_qubits: int) -> QuantumCircuit:
    """建立與第 16 課相同、包含最後 SWAP 的 QFT 電路。"""
    circuit = QuantumCircuit(number_of_qubits, name="QFT")

    for target in reversed(range(number_of_qubits)):
        circuit.h(target)

        for control in reversed(range(target)):
            distance = target - control
            circuit.cp(np.pi / (2**distance), control, target)

    for qubit in range(number_of_qubits // 2):
        circuit.swap(qubit, number_of_qubits - qubit - 1)

    return circuit


def phase_estimation_circuit(
    phase: float,
    number_of_counting_qubits: int,
) -> QuantumCircuit:
    """估計 P(2*pi*phase) 在 eigenstate |1> 上的 eigenphase。"""
    if not 0.0 <= phase < 1.0:
        raise ValueError("phase must be in [0, 1)")
    if number_of_counting_qubits < 1:
        raise ValueError("number_of_counting_qubits must be positive")

    target = number_of_counting_qubits
    circuit = QuantumCircuit(number_of_counting_qubits + 1)

    # 1. P(theta)|1> = exp(i*theta)|1>，所以 |1> 是 eigenstate。
    circuit.x(target)

    # 2. Counting register 準備為所有整數 k 的均勻疊加。
    circuit.h(range(number_of_counting_qubits))
    circuit.barrier()

    # 3. q_j 控制 U^(2^j)，把 exp(2*pi*i*phase*2^j) kick back。
    for counting_qubit in range(number_of_counting_qubits):
        power = 2**counting_qubit
        angle = 2.0 * np.pi * phase * power
        circuit.cp(angle, counting_qubit, target)
    circuit.barrier()

    # 4. 將 Fourier-basis phase pattern 解碼為 binary integer y。
    inverse_qft = qft_circuit(number_of_counting_qubits).inverse()
    circuit.compose(
        inverse_qft,
        qubits=range(number_of_counting_qubits),
        inplace=True,
    )

    return circuit


def circular_error(estimate: float, target: float) -> float:
    """相位以 1 為週期，0.99 與 0.01 的距離應為 0.02。"""
    direct_error = abs(estimate - target)
    return min(direct_error, 1.0 - direct_error)


def main() -> None:
    phase = 3.0 / 8.0  # binary fraction 0.011
    number_of_counting_qubits = 3
    shots = 2048
    counting_qubits = list(range(number_of_counting_qubits))

    circuit = phase_estimation_circuit(
        phase,
        number_of_counting_qubits,
    )
    state = Statevector.from_instruction(circuit)

    exact_probabilities = {
        str(outcome): float(probability)
        for outcome, probability in state.probabilities_dict(
            qargs=counting_qubits,
            decimals=12,
        ).items()
        if probability > 1e-12
    }

    state.seed(1700)
    counts = {
        str(outcome): int(count)
        for outcome, count in state.sample_counts(
            shots=shots,
            qargs=counting_qubits,
        ).items()
    }

    most_likely_bits = max(counts, key=counts.get)
    measured_integer = int(most_likely_bits, 2)
    estimated_phase = measured_integer / (2**number_of_counting_qubits)

    print("Quantum phase-estimation circuit:")
    print(circuit.draw(output="text"))
    print(f"\nTrue phase:             {phase:.6f}")
    print(f"Binary phase:           0.{most_likely_bits}")
    print(f"Exact probabilities:    {exact_probabilities}")
    print(f"Counts ({shots} shots):  {counts}")
    print(f"Measured integer y:     {measured_integer}")
    print(f"Estimated phase y/2^m:  {estimated_phase:.6f}")
    print(
        "Circular estimation error: "
        f"{circular_error(estimated_phase, phase):.3e}"
    )


if __name__ == "__main__":
    main()
