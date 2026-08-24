"""Day 06：量測、投影與量子態塌縮。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def main() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    bell = Statevector.from_instruction(circuit)

    print("Bell state:", np.round(bell.data, 6))
    print("Probabilities:", bell.probabilities_dict())
    print("\nSequential measurements:")

    # 每次都由同一 Bell state 重新開始，先量 q0，再量 q1。
    for seed in range(8):
        state = bell.copy()
        state.seed(seed)
        result_q0, collapsed = state.measure(qargs=[0])
        result_q1, _ = collapsed.measure(qargs=[1])
        print(
            f"seed={seed}: q0={result_q0}, q1={result_q1}, "
            f"collapsed={collapsed.probabilities_dict()}"
        )

    # Basis order: |00>, |01>, |10>, |11>。
    psi = bell.data
    p0_operator = np.diag([1, 0, 1, 0])
    p1_operator = np.diag([0, 1, 0, 1])

    probability_0 = np.vdot(psi, p0_operator @ psi).real
    probability_1 = np.vdot(psi, p1_operator @ psi).real
    state_after_0 = p0_operator @ psi / np.sqrt(probability_0)
    state_after_1 = p1_operator @ psi / np.sqrt(probability_1)

    print("\nManual projection:")
    print("P(q0=0) =", probability_0)
    print("P(q0=1) =", probability_1)
    print("State after q0=0:", np.round(state_after_0, 6))
    print("State after q0=1:", np.round(state_after_1, 6))


if __name__ == "__main__":
    main()

