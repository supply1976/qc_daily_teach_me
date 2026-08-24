"""Day 05：量子糾纏與 Bell state。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector, partial_trace


def main() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    state = Statevector.from_instruction(circuit)
    state.seed(42)

    print("Quantum circuit:")
    print(circuit.draw())
    print("\nStatevector:", np.round(state.data, 6))
    print("Probabilities:", state.probabilities_dict())
    print("1000 measurements:", state.sample_counts(shots=1000))

    # Trace out q1 留下 q0；trace out q0 留下 q1。
    rho_q0 = partial_trace(state, [1])
    rho_q1 = partial_trace(state, [0])

    print("\nReduced density matrix of q0:")
    print(np.round(rho_q0.data, 6))
    print("Reduced density matrix of q1:")
    print(np.round(rho_q1.data, 6))
    print("Global purity:", state.purity())
    print("q0 purity:", rho_q0.purity())
    print("q1 purity:", rho_q1.purity())

    print("<ZZ> =", np.real_if_close(state.expectation_value(Pauli("ZZ"))))
    print("<XX> =", np.real_if_close(state.expectation_value(Pauli("XX"))))


if __name__ == "__main__":
    main()

