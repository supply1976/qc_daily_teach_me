"""Day 04：兩個 qubit、張量積與 Qiskit 位元順序。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def main() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)  # q0 -> |+>
    circuit.x(1)  # q1 -> |1>

    state = Statevector.from_instruction(circuit)
    state.seed(42)

    print("Quantum circuit:")
    print(circuit.draw())
    print("\nQiskit statevector:")
    print(np.round(state.data, 6))
    print("\nBasis probabilities:")
    print(state.probabilities_dict())
    print("\n1000 measurements:")
    print(state.sample_counts(shots=1000))

    q0_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    q1_one = np.array([0, 1], dtype=complex)

    # Qiskit basis label 是 |q1 q0>，所以 q1 放在 kron 左側。
    manual_state = np.kron(q1_one, q0_plus)

    print("\nNumPy tensor product:")
    print(np.round(manual_state, 6))
    print("Qiskit and NumPy agree:", np.allclose(state.data, manual_state))
    print("q0 probabilities:", state.probabilities(qargs=[0]))
    print("q1 probabilities:", state.probabilities(qargs=[1]))


if __name__ == "__main__":
    main()

