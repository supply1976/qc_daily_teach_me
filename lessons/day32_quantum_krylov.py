"""Day 32: Hadamard-test matrix elements and quantum Krylov diagonalization."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector


IDENTITY = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def hamiltonian_matrix(z_field: float, x_field: float) -> np.ndarray:
    return z_field * PAULI_Z + x_field * PAULI_X


def evolution_matrix(
    time: float,
    z_field: float,
    x_field: float,
) -> np.ndarray:
    """Use H^2 = omega^2 I to evaluate exp(-i H t)."""
    hamiltonian = hamiltonian_matrix(z_field, x_field)
    omega = np.sqrt(z_field**2 + x_field**2)

    return (
        np.cos(omega * time) * IDENTITY
        - 1j * np.sin(omega * time) * hamiltonian / omega
    )


def transition_operator(
    left_time: float,
    right_time: float,
    observable: np.ndarray,
    z_field: float,
    x_field: float,
) -> np.ndarray:
    """Return U(left)^dagger observable U(right)."""
    left = evolution_matrix(left_time, z_field, x_field)
    right = evolution_matrix(right_time, z_field, x_field)
    return left.conj().T @ observable @ right


def hadamard_test_circuit(
    operator: np.ndarray,
    component: str,
) -> QuantumCircuit:
    """Measure <0|operator|0>; q0 is ancilla and q1 is system."""
    if component not in {"real", "imag"}:
        raise ValueError("component must be 'real' or 'imag'")

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.append(
        UnitaryGate(operator, label="W").control(1),
        [0, 1],
    )

    if component == "imag":
        circuit.sdg(0)

    circuit.h(0)
    return circuit


def sampled_matrix_element(
    operator: np.ndarray,
    shots: int,
    seed: int,
) -> complex:
    estimates = []

    for offset, component in enumerate(["real", "imag"]):
        state = Statevector.from_instruction(
            hadamard_test_circuit(operator, component)
        )
        state.seed(seed + offset)
        counts = state.sample_counts(shots=shots, qargs=[0])
        estimates.append(
            (counts.get("0", 0) - counts.get("1", 0)) / shots
        )

    return complex(estimates[0], estimates[1])


def exact_matrix_element(operator: np.ndarray) -> complex:
    # The system reference state is |0>.
    return complex(operator[0, 0])


def krylov_matrices(
    times: np.ndarray,
    z_field: float,
    x_field: float,
    shots: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct overlap S and projected Hamiltonian H_K."""
    dimension = times.size
    overlap = np.zeros((dimension, dimension), dtype=complex)
    projected = np.zeros((dimension, dimension), dtype=complex)

    for row in range(dimension):
        for column in range(row, dimension):
            operators = [
                transition_operator(
                    times[row],
                    times[column],
                    observable,
                    z_field,
                    x_field,
                )
                for observable in [IDENTITY, PAULI_Z, PAULI_X]
            ]

            if shots is None:
                values = [
                    exact_matrix_element(operator)
                    for operator in operators
                ]
            else:
                values = [
                    sampled_matrix_element(
                        operator,
                        shots,
                        seed + 20 * (row * dimension + column) + 2 * index,
                    )
                    for index, operator in enumerate(operators)
                ]

            overlap[row, column] = values[0]
            projected[row, column] = (
                z_field * values[1] + x_field * values[2]
            )

            if row != column:
                overlap[column, row] = np.conj(overlap[row, column])
                projected[column, row] = np.conj(
                    projected[row, column]
                )

    # Hermitian observables have real diagonal matrix elements.
    overlap[np.diag_indices(dimension)] = np.real(
        np.diag(overlap)
    )
    projected[np.diag_indices(dimension)] = np.real(
        np.diag(projected)
    )

    return overlap, projected


def solve_generalized_eigenproblem(
    overlap: np.ndarray,
    projected: np.ndarray,
    cutoff: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve H_K c = E S c using symmetric orthogonalization."""
    overlap_values, overlap_vectors = np.linalg.eigh(overlap)
    if np.min(overlap_values) <= cutoff:
        raise ValueError("Krylov overlap matrix is singular or indefinite")

    inverse_sqrt = (
        overlap_vectors
        @ np.diag(1.0 / np.sqrt(overlap_values))
        @ overlap_vectors.conj().T
    )
    effective = inverse_sqrt @ projected @ inverse_sqrt
    effective = 0.5 * (effective + effective.conj().T)

    energies, orthogonal_vectors = np.linalg.eigh(effective)
    coefficients = inverse_sqrt @ orthogonal_vectors
    return energies, coefficients, overlap_values


def physical_ritz_states(
    times: np.ndarray,
    coefficients: np.ndarray,
    z_field: float,
    x_field: float,
) -> np.ndarray:
    basis = np.column_stack([
        evolution_matrix(time, z_field, x_field)[:, 0]
        for time in times
    ])
    states = basis @ coefficients
    return states / np.linalg.norm(states, axis=0, keepdims=True)


def main() -> None:
    z_field = 0.7
    x_field = 1.1
    time_step = 0.8
    shots = 16384
    times = np.array([0.0, time_step])

    full_hamiltonian = hamiltonian_matrix(z_field, x_field)
    exact_energies, exact_states = np.linalg.eigh(full_hamiltonian)

    exact_overlap, exact_projected = krylov_matrices(
        times,
        z_field,
        x_field,
    )
    exact_ritz, _, exact_overlap_values = (
        solve_generalized_eigenproblem(
            exact_overlap,
            exact_projected,
        )
    )

    sampled_overlap, sampled_projected = krylov_matrices(
        times,
        z_field,
        x_field,
        shots=shots,
        seed=32000,
    )
    sampled_ritz, sampled_coefficients, sampled_overlap_values = (
        solve_generalized_eigenproblem(
            sampled_overlap,
            sampled_projected,
        )
    )

    ritz_states = physical_ritz_states(
        times,
        sampled_coefficients,
        z_field,
        x_field,
    )
    fidelities = np.array([
        abs(np.vdot(exact_states[:, index], ritz_states[:, index])) ** 2
        for index in range(2)
    ])

    print("Hamiltonian H = 0.7 Z + 1.1 X")
    print("Krylov states: |0>, exp(-i H tau)|0>")
    print(f"tau:   {time_step:.3f}")
    print(f"shots: {shots} per Hadamard-test component")

    print("\nExact overlap matrix S:")
    print(np.round(exact_overlap, 6))
    print("\nExact projected Hamiltonian H_K:")
    print(np.round(exact_projected, 6))

    print("\nSampled overlap matrix S:")
    print(np.round(sampled_overlap, 6))
    print("\nSampled projected Hamiltonian H_K:")
    print(np.round(sampled_projected, 6))

    print("\nOverlap eigenvalues and condition numbers:")
    print(
        "exact:  ",
        np.round(exact_overlap_values, 6),
        f"condition={np.linalg.cond(exact_overlap):.6f}",
    )
    print(
        "sampled:",
        np.round(sampled_overlap_values, 6),
        f"condition={np.linalg.cond(sampled_overlap):.6f}",
    )

    print("\nEnergy comparison:")
    print(" state       exact E       exact Krylov      sampled Krylov")
    print("------------------------------------------------------------")
    for index, label in enumerate(["ground", "excited"]):
        print(
            f" {label:7s}  {exact_energies[index]:+.9f}   "
            f"{exact_ritz[index]:+.9f}      "
            f"{sampled_ritz[index]:+.9f}"
        )

    print("\nSampled Ritz-state fidelities:")
    print(f"ground:  {fidelities[0]:.9f}")
    print(f"excited: {fidelities[1]:.9f}")

    print("\nWhy the Krylov time step matters:")
    print(" tau    overlap condition number")
    print("---------------------------------")
    for candidate in [0.05, 0.20, 0.80]:
        candidate_overlap, _ = krylov_matrices(
            np.array([0.0, candidate]),
            z_field,
            x_field,
        )
        print(
            f" {candidate:4.2f}       "
            f"{np.linalg.cond(candidate_overlap):12.6f}"
        )


if __name__ == "__main__":
    main()
