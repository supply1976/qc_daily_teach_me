"""Day 01：一個 qubit、疊加態與量測。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def main() -> None:
    # qubit 預設由 |0> 開始；H|0> = (|0> + |1>) / sqrt(2)。
    circuit = QuantumCircuit(1)
    circuit.h(0)

    state = Statevector.from_instruction(circuit)
    state.seed(42)

    print("Quantum circuit:")
    print(circuit.draw())
    print("\nStatevector:")
    print(state.data)
    print("\nExact probabilities:")
    print(state.probabilities_dict())
    print("\n1000 simulated measurements:")
    print(state.sample_counts(shots=1000))
    print("\nBorn rule |amplitude|^2:")
    print(np.abs(state.data) ** 2)


if __name__ == "__main__":
    main()

