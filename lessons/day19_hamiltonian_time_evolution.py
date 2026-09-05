"""第 19 課：Hamiltonian time evolution 與 Larmor precession。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector, state_fidelity


def exact_bloch_vector(omega: float, time: float) -> np.ndarray:
    """H=(omega/2)Z、初態 |+> 的 Bloch-vector 解析解。"""
    angle = omega * time
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def expectation(state: Statevector, label: str) -> float:
    """計算單 qubit Pauli expectation value。"""
    value = np.real_if_close(state.expectation_value(Pauli(label)))
    return float(value)


def main() -> None:
    omega = 1.0
    hamiltonian = SparsePauliOp.from_list([("Z", omega / 2.0)])
    initial_state = Statevector.from_label("+")

    times = [
        0.0,
        np.pi / 4,
        np.pi / 2,
        np.pi,
        3 * np.pi / 2,
        2 * np.pi,
    ]

    print("Hamiltonian H = (omega/2) Z, omega = 1")
    print("Initial state = |+>")
    print("\n time       <X>       <Y>       <Z>       <H>    error")
    print("------------------------------------------------------------")

    maximum_error = 0.0

    for time in times:
        evolution = PauliEvolutionGate(
            hamiltonian,
            time=time,
        )
        evolved_state = initial_state.evolve(evolution.definition)

        bloch = np.array([
            expectation(evolved_state, "X"),
            expectation(evolved_state, "Y"),
            expectation(evolved_state, "Z"),
        ])
        exact = exact_bloch_vector(omega, time)
        error = float(np.max(np.abs(bloch - exact)))
        maximum_error = max(maximum_error, error)

        energy = float(
            np.real_if_close(
                evolved_state.expectation_value(hamiltonian)
            )
        )

        print(
            f"{time:6.3f}  "
            f"{bloch[0]:9.6f} "
            f"{bloch[1]:9.6f} "
            f"{bloch[2]:9.6f} "
            f"{energy:9.6f} "
            f"{error:.1e}"
        )

    # H=(omega/2)Z 的 evolution 應等價於 Rz(omega*t)。
    test_time = 0.7
    test_evolution = PauliEvolutionGate(
        hamiltonian,
        time=test_time,
    )
    pauli_evolved = initial_state.evolve(test_evolution.definition)

    rz_circuit = QuantumCircuit(1)
    rz_circuit.rz(omega * test_time, 0)
    rz_evolved = initial_state.evolve(rz_circuit)

    print(f"\nMaximum Bloch-vector error: {maximum_error:.3e}")
    print(
        "Fidelity with Rz(omega*t) at t=0.7: "
        f"{state_fidelity(pauli_evolved, rz_evolved):.12f}"
    )

    final_evolution = PauliEvolutionGate(
        hamiltonian,
        time=2 * np.pi,
    )
    final_state = initial_state.evolve(final_evolution.definition)
    print(
        "Return fidelity after one period:      "
        f"{state_fidelity(initial_state, final_state):.12f}"
    )
    print("Final amplitudes after one period:")
    print(np.round(final_state.data, 6))


if __name__ == "__main__":
    main()
