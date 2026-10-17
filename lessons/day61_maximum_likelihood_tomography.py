"""Day 61: Physical qubit tomography by maximum likelihood."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector


def tetrahedral_vectors():
    """Four tetrahedral Bloch vectors for the qubit SIC-POVM."""
    return np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]) / np.sqrt(3.0)


def bloch_ket(vector):
    """Return a pure-state ket with the requested Bloch vector."""
    x_component, y_component, z_component = vector
    polar_angle = np.arccos(
        np.clip(z_component, -1.0, 1.0)
    )
    azimuth = np.arctan2(y_component, x_component)
    return np.array([
        np.cos(polar_angle / 2.0),
        np.exp(1j * azimuth) * np.sin(polar_angle / 2.0),
    ])


def naimark_unitary(vectors):
    """Complete the SIC isometry to a two-qubit unitary."""
    isometry = np.array([
        bloch_ket(vector).conj() / np.sqrt(2.0)
        for vector in vectors
    ])
    complement_projector = (
        np.eye(4) - isometry @ isometry.conj().T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        complement_projector
    )
    complement = eigenvectors[:, np.argsort(eigenvalues)[-2:]]
    return np.column_stack([isometry, complement])


def input_bloch_vector(polar_angle, azimuth):
    return np.array([
        np.sin(polar_angle) * np.cos(azimuth),
        np.sin(polar_angle) * np.sin(azimuth),
        np.cos(polar_angle),
    ])


def sic_circuit(polar_angle, azimuth, unitary):
    """Prepare a pure qubit and perform the SIC Naimark dilation."""
    circuit = QuantumCircuit(2)
    circuit.ry(float(polar_angle), 0)
    circuit.rz(float(azimuth), 0)
    circuit.append(UnitaryGate(unitary, label="SIC"), [0, 1])
    return circuit


def sic_probabilities(bloch_vector, vectors):
    """Born probabilities p_k = (1 + n_k dot r) / 4."""
    return (1.0 + vectors @ bloch_vector) / 4.0


def linear_inversion(frequencies, vectors):
    """Unconstrained inverse of the SIC probability map."""
    return 3.0 * np.asarray(frequencies) @ vectors


def project_to_bloch_ball(vector, radius=1.0):
    """Euclidean projection onto |r| <= radius."""
    norm = np.linalg.norm(vector)
    if norm <= radius:
        return vector.copy()
    return radius * vector / norm


def negative_log_likelihood(
    bloch_vector,
    frequencies,
    vectors,
):
    """Cross entropy between measured frequencies and the model."""
    probabilities = np.clip(
        sic_probabilities(bloch_vector, vectors),
        1e-15,
        None,
    )
    return -float(frequencies @ np.log(probabilities))


def maximum_likelihood_estimate(
    frequencies,
    vectors,
    max_iterations=500,
    tolerance=1e-11,
):
    """Projected-gradient MLE constrained to the Bloch ball."""
    estimate = np.zeros(3, dtype=float)
    physical_radius = 1.0 - 1e-12

    for iteration in range(max_iterations):
        probabilities = np.clip(
            sic_probabilities(estimate, vectors),
            1e-15,
            None,
        )
        gradient = -np.sum(
            (
                frequencies
                / (4.0 * probabilities)
            )[:, None]
            * vectors,
            axis=0,
        )

        current_loss = negative_log_likelihood(
            estimate,
            frequencies,
            vectors,
        )
        step_size = 1.0

        # Backtracking keeps every accepted step likelihood-improving.
        for _ in range(40):
            candidate = project_to_bloch_ball(
                estimate - step_size * gradient,
                physical_radius,
            )
            candidate_loss = negative_log_likelihood(
                candidate,
                frequencies,
                vectors,
            )
            if candidate_loss <= current_loss - 1e-13:
                break
            step_size *= 0.5

        displacement = np.linalg.norm(candidate - estimate)
        estimate = candidate
        if displacement < tolerance:
            break

    return estimate, iteration + 1


def density_eigenvalues(bloch_vector):
    """Eigenvalues of rho = (I + r dot sigma) / 2."""
    norm = np.linalg.norm(bloch_vector)
    return np.array([
        0.5 * (1.0 - norm),
        0.5 * (1.0 + norm),
    ])


def pure_state_fidelity(estimate, true_bloch):
    """F(rho_estimate, |psi_true>) for a pure true state."""
    return 0.5 * (1.0 + estimate @ true_bloch)


def root_mean_square_error(estimates, truth):
    squared_errors = np.sum(
        (estimates - truth) ** 2,
        axis=1,
    )
    return float(np.sqrt(np.mean(squared_errors)))


def main():
    polar_angle = 1.1
    azimuth = 0.7
    shots = 256
    repetitions = 1_000
    seed = 20261017

    vectors = tetrahedral_vectors()
    unitary = naimark_unitary(vectors)
    true_bloch = input_bloch_vector(
        polar_angle,
        azimuth,
    )
    exact_probabilities = sic_probabilities(
        true_bloch,
        vectors,
    )
    qiskit_probabilities = Statevector.from_instruction(
        sic_circuit(polar_angle, azimuth, unitary)
    ).probabilities()

    rng = np.random.default_rng(seed)

    # One illustrative experiment.
    counts = rng.multinomial(shots, exact_probabilities)
    frequencies = counts / shots
    linear_estimate = linear_inversion(
        frequencies,
        vectors,
    )
    radial_estimate = project_to_bloch_ball(
        linear_estimate
    )
    mle_estimate, mle_iterations = maximum_likelihood_estimate(
        frequencies,
        vectors,
    )

    # Repeat to compare finite-shot estimators statistically.
    repeated_counts = rng.multinomial(
        shots,
        exact_probabilities,
        size=repetitions,
    )
    repeated_frequencies = repeated_counts / shots
    linear_estimates = linear_inversion(
        repeated_frequencies,
        vectors,
    )
    radial_estimates = np.array([
        project_to_bloch_ball(estimate)
        for estimate in linear_estimates
    ])
    mle_estimates = np.empty_like(linear_estimates)
    mle_iteration_counts = np.empty(repetitions, dtype=int)
    for index, observed_frequencies in enumerate(
        repeated_frequencies
    ):
        (
            mle_estimates[index],
            mle_iteration_counts[index],
        ) = maximum_likelihood_estimate(
            observed_frequencies,
            vectors,
        )

    radial_losses = np.array([
        negative_log_likelihood(
            estimate,
            observed_frequencies,
            vectors,
        )
        for estimate, observed_frequencies in zip(
            radial_estimates,
            repeated_frequencies,
        )
    ])
    mle_losses = np.array([
        negative_log_likelihood(
            estimate,
            observed_frequencies,
            vectors,
        )
        for estimate, observed_frequencies in zip(
            mle_estimates,
            repeated_frequencies,
        )
    ])

    print("Maximum-likelihood SIC tomography")
    print(f"Shots per experiment:    {shots}")
    print(f"Monte Carlo repetitions: {repetitions}")
    print(
        "Maximum Qiskit probability error: "
        f"{np.max(np.abs(qiskit_probabilities - exact_probabilities)):.3e}"
    )
    print(
        "True Bloch vector: "
        + np.array2string(true_bloch, precision=9)
    )
    print(
        "Exact probabilities: "
        + np.array2string(exact_probabilities, precision=9)
    )

    print("\nOne finite-shot data set")
    print("Counts:      " + np.array2string(counts))
    print(
        "Frequencies: "
        + np.array2string(frequencies, precision=9)
    )
    print(
        "Linear inversion: "
        + np.array2string(linear_estimate, precision=9)
    )
    print(
        f"  |r| = {np.linalg.norm(linear_estimate):.9f}"
    )
    print(
        "  density eigenvalues = "
        + np.array2string(
            density_eigenvalues(linear_estimate),
            precision=9,
        )
    )
    print(
        "Radial projection: "
        + np.array2string(radial_estimate, precision=9)
    )
    print(
        "  normalized NLL = "
        f"{negative_log_likelihood(radial_estimate, frequencies, vectors):.12f}"
    )
    print(
        "Maximum likelihood: "
        + np.array2string(mle_estimate, precision=9)
    )
    print(f"  iterations = {mle_iterations}")
    print(
        f"  |r| = {np.linalg.norm(mle_estimate):.12f}"
    )
    print(
        "  density eigenvalues = "
        + np.array2string(
            density_eigenvalues(mle_estimate),
            precision=9,
        )
    )
    print(
        "  normalized NLL = "
        f"{negative_log_likelihood(mle_estimate, frequencies, vectors):.12f}"
    )

    linear_norms = np.linalg.norm(
        linear_estimates,
        axis=1,
    )
    radial_fidelities = np.array([
        pure_state_fidelity(estimate, true_bloch)
        for estimate in radial_estimates
    ])
    mle_fidelities = np.array([
        pure_state_fidelity(estimate, true_bloch)
        for estimate in mle_estimates
    ])

    print("\nMonte Carlo comparison")
    print(
        "Unphysical linear estimates: "
        f"{np.mean(linear_norms > 1.0):.3%}"
    )
    print(
        "Linear-inversion Bloch RMSE: "
        f"{root_mean_square_error(linear_estimates, true_bloch):.9f}"
    )
    print(
        "Radial-projection Bloch RMSE: "
        f"{root_mean_square_error(radial_estimates, true_bloch):.9f}"
    )
    print(
        "Maximum-likelihood Bloch RMSE: "
        f"{root_mean_square_error(mle_estimates, true_bloch):.9f}"
    )
    print(
        "Mean radial-projection fidelity: "
        f"{np.mean(radial_fidelities):.9f}"
    )
    print(
        "Mean maximum-likelihood fidelity: "
        f"{np.mean(mle_fidelities):.9f}"
    )
    print(
        "Mean radial-projection NLL: "
        f"{np.mean(radial_losses):.12f}"
    )
    print(
        "Mean maximum-likelihood NLL: "
        f"{np.mean(mle_losses):.12f}"
    )
    print(
        "MLE likelihood >= radial likelihood: "
        f"{np.mean(mle_losses <= radial_losses + 1e-12):.3%}"
    )
    print(
        "Mean MLE optimizer iterations: "
        f"{np.mean(mle_iteration_counts):.3f}"
    )


if __name__ == "__main__":
    main()
