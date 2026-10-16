"""Day 60: Qubit SIC-POVM tomography through a Naimark dilation."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector


def tetrahedral_vectors():
    """The four vertices of a regular tetrahedron on the Bloch sphere."""
    return np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]) / np.sqrt(3.0)


def bloch_ket(vector):
    """Return a pure qubit ket whose Bloch vector is ``vector``."""
    x_component, y_component, z_component = vector
    polar_angle = np.arccos(
        np.clip(z_component, -1.0, 1.0)
    )
    azimuth = np.arctan2(y_component, x_component)
    return np.array([
        np.cos(polar_angle / 2.0),
        np.exp(1j * azimuth) * np.sin(polar_angle / 2.0),
    ])


def sic_effects(vectors):
    """Build E_k = (I + n_k dot sigma) / 4."""
    effects = []
    for vector in vectors:
        ket = bloch_ket(vector)
        effects.append(
            0.5 * np.outer(ket, ket.conj())
        )
    return np.array(effects)


def naimark_unitary(vectors):
    """Embed the four-outcome POVM into a two-qubit unitary.

    The first two columns form an isometry V.  Starting with the
    ancilla in |0>, computational-basis outcome k has amplitude
    <phi_k|psi>/sqrt(2), and hence probability <psi|E_k|psi>.
    """
    isometry = np.array([
        bloch_ket(vector).conj() / np.sqrt(2.0)
        for vector in vectors
    ])

    # Complete V to a 4 x 4 unitary with an orthonormal complement.
    projector_onto_complement = (
        np.eye(4) - isometry @ isometry.conj().T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        projector_onto_complement
    )
    complement = eigenvectors[:, np.argsort(eigenvalues)[-2:]]
    return np.column_stack([isometry, complement])


def input_bloch_vector(polar_angle, azimuth):
    """Bloch vector prepared by Rz(azimuth) Ry(polar_angle)|0>."""
    return np.array([
        np.sin(polar_angle) * np.cos(azimuth),
        np.sin(polar_angle) * np.sin(azimuth),
        np.cos(polar_angle),
    ])


def sic_circuit(polar_angle, azimuth, unitary):
    """Prepare one qubit and apply a two-qubit Naimark dilation."""
    circuit = QuantumCircuit(2)
    circuit.ry(float(polar_angle), 0)
    circuit.rz(float(azimuth), 0)
    # q0 is the system and q1 begins as the dilation ancilla |0>.
    circuit.append(UnitaryGate(unitary, label="SIC"), [0, 1])
    return circuit


def analytic_probabilities(bloch_vector, vectors):
    """p_k = (1 + r dot n_k) / 4."""
    return (1.0 + vectors @ bloch_vector) / 4.0


def reconstruct_bloch(probabilities, vectors):
    """Linear SIC inversion: r = 3 sum_k p_k n_k."""
    return 3.0 * np.asarray(probabilities) @ vectors


def project_to_bloch_ball(estimates):
    """Radially project unphysical linear estimates into |r| <= 1."""
    norms = np.linalg.norm(estimates, axis=1)
    scales = np.maximum(norms, 1.0)
    return estimates / scales[:, None]


def sic_monte_carlo(
    probabilities,
    vectors,
    shots,
    repetitions,
    rng,
):
    counts = rng.multinomial(
        shots,
        probabilities,
        size=repetitions,
    )
    frequencies = counts / shots
    return reconstruct_bloch(frequencies, vectors)


def pauli_monte_carlo(
    bloch_vector,
    total_shots,
    repetitions,
    rng,
):
    """Standard tomography with the same total copies split over X/Y/Z."""
    shots_per_axis = np.full(3, total_shots // 3, dtype=int)
    shots_per_axis[: total_shots % 3] += 1
    estimates = np.empty((repetitions, 3), dtype=float)
    for axis in range(3):
        plus_probability = 0.5 * (
            1.0 + bloch_vector[axis]
        )
        plus_counts = rng.binomial(
            shots_per_axis[axis],
            plus_probability,
            size=repetitions,
        )
        estimates[:, axis] = (
            2.0 * plus_counts / shots_per_axis[axis] - 1.0
        )
    return estimates, shots_per_axis


def root_mean_square_vector_error(estimates, truth):
    squared_errors = np.sum(
        (estimates - truth) ** 2,
        axis=1,
    )
    return float(np.sqrt(np.mean(squared_errors)))


def main():
    polar_angle = 1.1
    azimuth = 0.7
    shots = 4_096
    repetitions = 5_000
    seed = 20261016

    vectors = tetrahedral_vectors()
    effects = sic_effects(vectors)
    unitary = naimark_unitary(vectors)
    true_bloch = input_bloch_vector(
        polar_angle,
        azimuth,
    )
    exact_probabilities = analytic_probabilities(
        true_bloch,
        vectors,
    )

    state = Statevector.from_instruction(
        sic_circuit(polar_angle, azimuth, unitary)
    )
    qiskit_probabilities = state.probabilities()
    exact_reconstruction = reconstruct_bloch(
        qiskit_probabilities,
        vectors,
    )

    effect_sum_error = np.max(
        np.abs(np.sum(effects, axis=0) - np.eye(2))
    )
    overlap_matrix = np.array([
        [
            np.real(np.trace(first @ second))
            for second in effects
        ]
        for first in effects
    ])
    off_diagonal_mask = ~np.eye(4, dtype=bool)
    unitarity_error = np.max(
        np.abs(unitary.conj().T @ unitary - np.eye(4))
    )

    rng = np.random.default_rng(seed)
    sic_estimates = sic_monte_carlo(
        exact_probabilities,
        vectors,
        shots,
        repetitions,
        rng,
    )
    pauli_estimates, shots_per_axis = pauli_monte_carlo(
        true_bloch,
        shots,
        repetitions,
        rng,
    )

    sic_rmse = root_mean_square_vector_error(
        sic_estimates,
        true_bloch,
    )
    pauli_rmse = root_mean_square_vector_error(
        pauli_estimates,
        true_bloch,
    )
    predicted_sic_rmse = np.sqrt(
        (9.0 - np.dot(true_bloch, true_bloch)) / shots
    )
    predicted_pauli_rmse = np.sqrt(
        np.sum(
            (1.0 - true_bloch**2) / shots_per_axis
        )
    )

    sic_norms = np.linalg.norm(sic_estimates, axis=1)
    projected_estimates = project_to_bloch_ball(sic_estimates)
    projected_rmse = root_mean_square_vector_error(
        projected_estimates,
        true_bloch,
    )
    projected_fidelities = 0.5 * (
        1.0 + projected_estimates @ true_bloch
    )

    print("Qubit tetrahedral SIC-POVM")
    print("Tetrahedral Bloch vectors:")
    print(np.array2string(vectors, precision=6))
    print(
        "Maximum POVM completeness error: "
        f"{effect_sum_error:.3e}"
    )
    print(
        "Diagonal Tr(E_k^2):              "
        f"{np.diag(overlap_matrix)[0]:.9f}"
    )
    print(
        "Maximum off-diagonal Tr(E_j E_k): "
        f"{np.max(overlap_matrix[off_diagonal_mask]):.9f}"
    )
    print(
        "Naimark-unitary error:            "
        f"{unitarity_error:.3e}"
    )

    print("\nInput pure state")
    print(
        "True Bloch vector: "
        + np.array2string(true_bloch, precision=9)
    )
    print(
        "Analytic SIC probabilities: "
        + np.array2string(exact_probabilities, precision=9)
    )
    print(
        "Qiskit SIC probabilities:  "
        + np.array2string(qiskit_probabilities, precision=9)
    )
    print(
        "Maximum probability error: "
        f"{np.max(np.abs(qiskit_probabilities - exact_probabilities)):.3e}"
    )
    print(
        "Exact reconstructed Bloch vector: "
        + np.array2string(exact_reconstruction, precision=9)
    )
    print(
        "Exact reconstruction error:       "
        f"{np.linalg.norm(exact_reconstruction - true_bloch):.3e}"
    )

    print("\nFinite-shot tomography")
    print(f"Total copies per trial: {shots}")
    print(f"Monte Carlo trials:     {repetitions}")
    print("SIC: one fixed four-outcome setting")
    print(f"  predicted Bloch RMSE: {predicted_sic_rmse:.9f}")
    print(f"  observed Bloch RMSE:  {sic_rmse:.9f}")
    print(
        "  unphysical linear estimates: "
        f"{np.mean(sic_norms > 1.0):.3%}"
    )
    print(f"  projected Bloch RMSE: {projected_rmse:.9f}")
    print(
        "  mean projected fidelity: "
        f"{np.mean(projected_fidelities):.9f}"
    )
    print("Pauli: copies split over X/Y/Z settings")
    print(
        "  shots per axis: "
        + np.array2string(shots_per_axis)
    )
    print(f"  predicted Bloch RMSE: {predicted_pauli_rmse:.9f}")
    print(f"  observed Bloch RMSE:  {pauli_rmse:.9f}")


if __name__ == "__main__":
    main()
