"""第 20 課：非對易 Hamiltonian 與 Lie–Trotter product formula。"""

import numpy as np
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector, state_fidelity
from qiskit.synthesis import LieTrotter


def exact_evolution(
    initial_state: Statevector,
    hamiltonian: SparsePauliOp,
    time: float,
) -> Statevector:
    """用 NumPy 對角化 H，計算精確的 exp(-iHt)|psi>。"""
    matrix = hamiltonian.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    phases = np.exp(-1j * eigenvalues * time)
    unitary = eigenvectors @ np.diag(phases) @ eigenvectors.conj().T
    return Statevector(unitary @ initial_state.data)


def trotter_evolution(
    initial_state: Statevector,
    hamiltonian: SparsePauliOp,
    time: float,
    repetitions: int,
) -> tuple[Statevector, int, dict[str, int]]:
    """用 r 個 Lie–Trotter 時間切片近似 exp(-iHt)。"""
    synthesis = LieTrotter(reps=repetitions)
    gate = PauliEvolutionGate(
        hamiltonian,
        time=time,
        synthesis=synthesis,
    )
    circuit = gate.definition
    state = initial_state.evolve(circuit)
    gate_counts = {
        str(name): int(count)
        for name, count in circuit.count_ops().items()
    }
    return state, circuit.depth(), gate_counts


def expectation_z(state: Statevector) -> float:
    value = np.real_if_close(state.expectation_value(Pauli("Z")))
    return float(value)


def main() -> None:
    # XZ = -ZX，因此這兩個 Hamiltonian terms 不對易。
    hamiltonian = SparsePauliOp.from_list([
        ("X", 1.0),
        ("Z", 1.0),
    ])
    initial_state = Statevector.from_label("0")
    evolution_time = 1.0

    exact_state = exact_evolution(
        initial_state,
        hamiltonian,
        evolution_time,
    )
    exact_z = expectation_z(exact_state)

    commutator = (
        Pauli("X").to_matrix() @ Pauli("Z").to_matrix()
        - Pauli("Z").to_matrix() @ Pauli("X").to_matrix()
    )

    print("Hamiltonian H = X + Z")
    print(f"||[X,Z]||_F = {np.linalg.norm(commutator):.6f}")
    print(f"Evolution time t = {evolution_time}")
    print(f"Exact <Z> = {exact_z:.9f}")
    print("\n reps  depth   fidelity       infidelity       <Z>       |Z error|")
    print("-------------------------------------------------------------------")

    for repetitions in [1, 2, 4, 8, 16, 32]:
        state, depth, gate_counts = trotter_evolution(
            initial_state,
            hamiltonian,
            evolution_time,
            repetitions,
        )
        fidelity = float(state_fidelity(exact_state, state))
        z_value = expectation_z(state)

        print(
            f"{repetitions:5d} "
            f"{depth:6d}   "
            f"{fidelity:.9f}   "
            f"{1.0 - fidelity:.3e}   "
            f"{z_value: .6f}   "
            f"{abs(z_value - exact_z):.3e}"
        )

        if repetitions in (1, 4):
            print(f"      gate counts: {gate_counts}")

    print("\nExact state amplitudes:")
    print(np.round(exact_state.data, 6))


if __name__ == "__main__":
    main()
