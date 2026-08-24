"""Day 02：相位與量子干涉。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def analyze(name: str, gates: list[str], shots: int = 100) -> None:
    circuit = QuantumCircuit(1)

    for gate in gates:
        if gate == "H":
            circuit.h(0)
        elif gate == "Z":
            circuit.z(0)
        else:
            raise ValueError(f"Unknown gate: {gate}")

    state = Statevector.from_instruction(circuit)
    state.seed(42)

    print(f"\n--- {name} ---")
    print(circuit.draw())
    print("amplitudes  [alpha, beta] =", np.round(state.data, 10))
    print("probability [P(0), P(1)] =", np.round(np.abs(state.data) ** 2, 10))
    print("measurement counts       =", state.sample_counts(shots=shots))


def main() -> None:
    analyze("H |0>", ["H"])
    analyze("H H |0>", ["H", "H"])
    analyze("H Z H |0>", ["H", "Z", "H"])


if __name__ == "__main__":
    main()

