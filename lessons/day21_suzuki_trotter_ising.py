"""第 21 課：二階 Suzuki–Trotter 與兩 qubit Ising dynamics。"""

import numpy as np
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import (
    SparsePauliOp,
    Statevector,
    entropy,
    partial_trace,
    state_fidelity,
)
from qiskit.synthesis import LieTrotter, SuzukiTrotter


def exact_evolution(
    initial_state: Statevector,
    hamiltonian: SparsePauliOp,
    time: float,
) -> Statevector:
    """用 NumPy diagonalization 計算精確 exp(-iHt)|psi>。"""
    matrix = hamiltonian.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    unitary = (
        eigenvectors
        @ np.diag(np.exp(-1j * eigenvalues * time))
        @ eigenvectors.conj().T
    )
    return Statevector(unitary @ initial_state.data)


def approximate_evolution(
    initial_state: Statevector,
    hamiltonian: SparsePauliOp,
    time: float,
    synthesis: LieTrotter | SuzukiTrotter,
) -> tuple[Statevector, int, dict[str, int]]:
    """合成 product-formula circuit 並回傳 state、depth 與 gate counts。"""
    gate = PauliEvolutionGate(
        hamiltonian,
        time=time,
        synthesis=synthesis,
    )
    circuit = gate.definition
    state = initial_state.evolve(circuit)
    counts = {
        str(name): int(count)
        for name, count in circuit.decompose(reps=2).count_ops().items()
    }
    return state, circuit.decompose(reps=2).depth(), counts


def energy(state: Statevector, hamiltonian: SparsePauliOp) -> float:
    value = np.real_if_close(state.expectation_value(hamiltonian))
    return float(value)


def entanglement_entropy(state: Statevector) -> float:
    """對 pure two-qubit state，單 qubit entropy 就是 entanglement entropy。"""
    reduced_q0 = partial_trace(state, [1])
    return float(entropy(reduced_q0, base=2))


def main() -> None:
    coupling = 0.8
    field = 1.1
    evolution_time = 0.5

    # Qiskit label 採 |q1 q0>：IX 是 X on q0，XI 是 X on q1。
    hamiltonian = SparsePauliOp.from_list([
        ("ZZ", -coupling),
        ("IX", -field),
        ("XI", -field),
    ])
    initial_state = Statevector.from_label("00")
    initial_energy = energy(initial_state, hamiltonian)

    exact_state = exact_evolution(
        initial_state,
        hamiltonian,
        evolution_time,
    )
    exact_entropy = entanglement_entropy(exact_state)

    print("Two-qubit transverse-field Ising model")
    print(f"H = -{coupling} ZZ - {field} (IX + XI)")
    print(f"Evolution time: {evolution_time}")
    print(f"Conserved exact energy: {initial_energy:.9f}")
    print(f"Exact entanglement entropy: {exact_entropy:.9f} bit")
    print(
        "\n method   reps  depth    fidelity      "
        "|energy drift|   entropy   |S error|"
    )
    print("---------------------------------------------------------------------")

    for repetitions in [1, 2, 4, 8]:
        methods = [
            ("Lie-1", LieTrotter(reps=repetitions)),
            ("Suzuki-2", SuzukiTrotter(order=2, reps=repetitions)),
        ]

        for name, synthesis in methods:
            state, depth, counts = approximate_evolution(
                initial_state,
                hamiltonian,
                evolution_time,
                synthesis,
            )
            fidelity = float(state_fidelity(exact_state, state))
            energy_drift = abs(energy(state, hamiltonian) - initial_energy)
            state_entropy = entanglement_entropy(state)

            print(
                f"{name:9s} {repetitions:4d} "
                f"{depth:6d}   {fidelity:.9f}   "
                f"{energy_drift:.3e}      "
                f"{state_entropy:.6f}   "
                f"{abs(state_entropy - exact_entropy):.3e}"
            )

            if repetitions == 1:
                print(f"           gate counts: {counts}")

    print("\nExact final probabilities:")
    print({
        str(key): round(float(value), 6)
        for key, value in exact_state.probabilities_dict().items()
        if value > 1e-12
    })


if __name__ == "__main__":
    main()
