"""Day 33: Regularize a noisy quantum Krylov eigenproblem."""

import numpy as np

from day32_quantum_krylov import (
    hamiltonian_matrix,
    krylov_matrices,
    physical_ritz_states,
)


def truncated_generalized_eigensolver(
    overlap: np.ndarray,
    projected: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonical orthogonalization after removing small S eigenvalues."""
    overlap = 0.5 * (overlap + overlap.conj().T)
    projected = 0.5 * (projected + projected.conj().T)

    overlap_values, overlap_vectors = np.linalg.eigh(overlap)
    keep = overlap_values > cutoff

    if not np.any(keep):
        raise ValueError("cutoff removed the complete Krylov space")

    orthogonalizer = (
        overlap_vectors[:, keep]
        @ np.diag(1.0 / np.sqrt(overlap_values[keep]))
    )

    effective_hamiltonian = (
        orthogonalizer.conj().T
        @ projected
        @ orthogonalizer
    )
    effective_hamiltonian = 0.5 * (
        effective_hamiltonian
        + effective_hamiltonian.conj().T
    )

    energies, orthogonal_vectors = np.linalg.eigh(
        effective_hamiltonian
    )
    coefficients = orthogonalizer @ orthogonal_vectors

    return energies, coefficients, overlap_values, keep


def state_fidelities(
    times: np.ndarray,
    coefficients: np.ndarray,
    exact_states: np.ndarray,
    z_field: float,
    x_field: float,
) -> np.ndarray:
    ritz_states = physical_ritz_states(
        times,
        coefficients,
        z_field,
        x_field,
    )

    return np.array([
        abs(np.vdot(exact_states[:, index], ritz_states[:, index])) ** 2
        for index in range(ritz_states.shape[1])
    ])


def main() -> None:
    z_field = 0.7
    x_field = 1.1
    time_step = 0.8
    shots = 4096

    # Three Krylov vectors live in a two-dimensional single-qubit space,
    # so one direction must be exactly redundant before sampling noise.
    times = np.array([0.0, time_step, 2.0 * time_step])

    full_hamiltonian = hamiltonian_matrix(z_field, x_field)
    exact_energies, exact_states = np.linalg.eigh(full_hamiltonian)

    exact_overlap, exact_projected = krylov_matrices(
        times,
        z_field,
        x_field,
    )
    sampled_overlap, sampled_projected = krylov_matrices(
        times,
        z_field,
        x_field,
        shots=shots,
        seed=33000,
    )

    exact_ritz, _, exact_overlap_values, exact_keep = (
        truncated_generalized_eigensolver(
            exact_overlap,
            exact_projected,
            cutoff=1e-8,
        )
    )

    # This threshold is too small: shot noise turns the null direction
    # into a small positive eigenvalue, so it is mistakenly retained.
    naive_ritz, _, sampled_overlap_values, naive_keep = (
        truncated_generalized_eigensolver(
            sampled_overlap,
            sampled_projected,
            cutoff=1e-8,
        )
    )

    regularized_ritz, regularized_coefficients, _, regularized_keep = (
        truncated_generalized_eigensolver(
            sampled_overlap,
            sampled_projected,
            cutoff=1e-2,
        )
    )
    fidelities = state_fidelities(
        times,
        regularized_coefficients,
        exact_states,
        z_field,
        x_field,
    )

    print("Hamiltonian H = 0.7 Z + 1.1 X")
    print("Krylov times:", times)
    print(f"Shots per Hadamard-test component: {shots}")
    print("Physical Hilbert-space dimension: 2")
    print("Number of Krylov vectors:         3")

    print("\nOverlap eigenvalues:")
    print("exact:  ", np.round(exact_overlap_values, 9))
    print("sampled:", np.round(sampled_overlap_values, 9))

    print("\nRetained overlap directions:")
    print(f"exact cutoff 1e-8:   {np.count_nonzero(exact_keep)}")
    print(f"sampled cutoff 1e-8: {np.count_nonzero(naive_keep)}")
    print(f"sampled cutoff 1e-2: {np.count_nonzero(regularized_keep)}")

    print("\nEnergy comparison:")
    print("exact full Hamiltonian:  ", np.round(exact_energies, 9))
    print("exact Krylov + cutoff:   ", np.round(exact_ritz, 9))
    print("sampled, cutoff 1e-8:   ", np.round(naive_ritz, 9))
    print("sampled, cutoff 1e-2:   ", np.round(regularized_ritz, 9))

    print("\nRegularized Ritz-state fidelities:")
    print(f"ground:  {fidelities[0]:.9f}")
    print(f"excited: {fidelities[1]:.9f}")

    print("\nCutoff sweep:")
    print(" cutoff     retained     energies")
    print("------------------------------------------------")
    for cutoff in [1e-8, 1e-3, 5e-3, 1e-2, 1e-1, 1.0]:
        energies, _, _, keep = truncated_generalized_eigensolver(
            sampled_overlap,
            sampled_projected,
            cutoff=cutoff,
        )
        formatted = np.array2string(
            energies,
            precision=6,
            floatmode="fixed",
        )
        print(
            f" {cutoff:8.1e}      {np.count_nonzero(keep):d}       "
            f"{formatted}"
        )


if __name__ == "__main__":
    main()
