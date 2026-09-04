"""第 18 課：QPE 作為 spectral measurement 與 eigenstate projection。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Pauli,
    Statevector,
    entropy,
    partial_trace,
)


def qft_circuit(number_of_qubits: int) -> QuantumCircuit:
    """建立包含最後 SWAP 的 QFT 電路。"""
    circuit = QuantumCircuit(number_of_qubits, name="QFT")

    for target in reversed(range(number_of_qubits)):
        circuit.h(target)
        for control in reversed(range(target)):
            distance = target - control
            circuit.cp(np.pi / (2**distance), control, target)

    for qubit in range(number_of_qubits // 2):
        circuit.swap(qubit, number_of_qubits - qubit - 1)

    return circuit


def spectral_qpe_circuit(
    phase: float,
    number_of_counting_qubits: int,
) -> QuantumCircuit:
    """對 target 的 |+> 執行 QPE，分辨 P gate 的兩個 eigenphases。"""
    target = number_of_counting_qubits
    circuit = QuantumCircuit(number_of_counting_qubits + 1)

    # |+> = (|0> + |1>)/sqrt(2)，是兩個 eigenstates 的疊加。
    circuit.h(target)
    circuit.h(range(number_of_counting_qubits))
    circuit.barrier()

    # P(2*pi*phase) 的 eigenphases：|0> -> 0，|1> -> phase。
    for counting_qubit in range(number_of_counting_qubits):
        power = 2**counting_qubit
        angle = 2.0 * np.pi * phase * power
        circuit.cp(angle, counting_qubit, target)
    circuit.barrier()

    circuit.compose(
        qft_circuit(number_of_counting_qubits).inverse(),
        qubits=range(number_of_counting_qubits),
        inplace=True,
    )

    return circuit


def conditional_target_state(
    joint_state: Statevector,
    measured_integer: int,
    number_of_counting_qubits: int,
) -> tuple[float, Statevector]:
    """固定 counting outcome，取出 target 的 normalized conditional state。"""
    dimension = 2**number_of_counting_qubits

    # Qiskit 排列為 |target, counting_(m-1), ..., counting_0>。
    tensor = joint_state.data.reshape(2, dimension)
    branch = tensor[:, measured_integer]
    probability = float(np.vdot(branch, branch).real)

    return probability, Statevector(branch / np.sqrt(probability))


def main() -> None:
    phase = 3.0 / 8.0
    number_of_counting_qubits = 3
    counting_qubits = list(range(number_of_counting_qubits))
    shots = 4096

    circuit = spectral_qpe_circuit(
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

    state.seed(1800)
    counts = {
        str(outcome): int(count)
        for outcome, count in state.sample_counts(
            shots=shots,
            qargs=counting_qubits,
        ).items()
    }

    # 消去 counting register，觀察尚未讀取 phase 時的 target。
    reduced_target = partial_trace(state, counting_qubits)
    target_entropy = float(entropy(reduced_target, base=2))

    print("QPE with a superposed target state:")
    print(circuit.draw(output="text"))
    print(f"\nNonzero exact phase probabilities: {exact_probabilities}")
    print(f"Counts ({shots} shots): {counts}")
    print("\nReduced target density matrix:")
    print(np.round(reduced_target.data, 6))
    print(f"Target entropy: {target_entropy:.6f} bit")

    print("\nConditional target states:")
    for bits in ["000", "011"]:
        measured_integer = int(bits, 2)
        probability, target_state = conditional_target_state(
            state,
            measured_integer,
            number_of_counting_qubits,
        )
        expectation_z = float(
            np.real_if_close(
                target_state.expectation_value(Pauli("Z"))
            )
        )

        print(
            f"  phase bits={bits}, probability={probability:.6f}, "
            f"target={np.round(target_state.data, 6)}, "
            f"<Z>={expectation_z:+.1f}"
        )


if __name__ == "__main__":
    main()
