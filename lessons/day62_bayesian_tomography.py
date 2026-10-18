"""Day 62: Bayesian qubit tomography with SIC measurements."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


def tetrahedral_vectors():
    """Four tetrahedral Bloch vectors for a qubit SIC-POVM."""
    return np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]) / np.sqrt(3.0)


def prepared_state(polar_angle, azimuth):
    """Prepare Rz(azimuth) Ry(polar_angle)|0> with Qiskit."""
    circuit = QuantumCircuit(1)
    circuit.ry(float(polar_angle), 0)
    circuit.rz(float(azimuth), 0)
    return Statevector.from_instruction(circuit)


def state_bloch_vector(state):
    """Read the X, Y, and Z expectations from a Qiskit state."""
    return np.array([
        float(np.real(state.expectation_value(Pauli(axis))))
        for axis in ("X", "Y", "Z")
    ])


def sic_probabilities(bloch_vectors, tetrahedron):
    """Return p_k = (1 + r dot n_k) / 4 for one or many states."""
    vectors = np.asarray(bloch_vectors)
    return (1.0 + vectors @ tetrahedron.T) / 4.0


def sample_uniform_bloch_ball(particle_count, rng):
    """Hilbert-Schmidt prior: uniform volume inside the Bloch ball."""
    directions = rng.normal(size=(particle_count, 3))
    directions /= np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )

    # A uniform three-dimensional volume has CDF P(R < r) = r^3.
    radii = rng.random(particle_count) ** (1.0 / 3.0)
    return directions * radii[:, None]


def posterior_weights(particles, counts, tetrahedron):
    """Importance weights proportional to prior times likelihood."""
    probabilities = np.clip(
        sic_probabilities(particles, tetrahedron),
        1e-15,
        None,
    )
    log_weights = np.log(probabilities) @ counts
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    return weights, probabilities


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    ordered_values = values[order]
    cumulative_weights = np.cumsum(weights[order])
    return np.interp(
        quantiles,
        cumulative_weights,
        ordered_values,
    )


def posterior_statistics(particles, weights):
    mean = weights @ particles
    centered = particles - mean
    covariance = centered.T @ (
        weights[:, None] * centered
    )
    intervals = np.array([
        weighted_quantile(
            particles[:, axis],
            weights,
            [0.05, 0.50, 0.95],
        )
        for axis in range(3)
    ])
    radius_interval = weighted_quantile(
        np.linalg.norm(particles, axis=1),
        weights,
        [0.05, 0.50, 0.95],
    )
    return mean, covariance, intervals, radius_interval


def effective_sample_size(weights):
    return float(1.0 / np.sum(weights**2))


def information_gain_bits(weights):
    """Discrete approximation to KL(posterior || prior)."""
    positive = weights > 0.0
    particle_count = weights.size
    return float(
        np.sum(
            weights[positive]
            * np.log2(weights[positive] * particle_count)
        )
    )


def linear_inversion(frequencies, tetrahedron):
    return 3.0 * frequencies @ tetrahedron


def project_to_bloch_ball(vector, radius=1.0):
    norm = np.linalg.norm(vector)
    if norm <= radius:
        return vector.copy()
    return radius * vector / norm


def negative_log_likelihood(
    bloch_vector,
    frequencies,
    tetrahedron,
):
    probabilities = np.clip(
        sic_probabilities(bloch_vector, tetrahedron),
        1e-15,
        None,
    )
    return -float(frequencies @ np.log(probabilities))


def maximum_likelihood_estimate(
    frequencies,
    tetrahedron,
    max_iterations=500,
):
    """Projected-gradient MLE from Day 61, included for comparison."""
    estimate = np.zeros(3)
    for _ in range(max_iterations):
        probabilities = np.clip(
            sic_probabilities(estimate, tetrahedron),
            1e-15,
            None,
        )
        gradient = -np.sum(
            (
                frequencies / (4.0 * probabilities)
            )[:, None]
            * tetrahedron,
            axis=0,
        )
        current_loss = negative_log_likelihood(
            estimate,
            frequencies,
            tetrahedron,
        )
        step_size = 1.0
        for _ in range(40):
            candidate = project_to_bloch_ball(
                estimate - step_size * gradient,
                1.0 - 1e-12,
            )
            if negative_log_likelihood(
                candidate,
                frequencies,
                tetrahedron,
            ) <= current_loss - 1e-13:
                break
            step_size *= 0.5

        if np.linalg.norm(candidate - estimate) < 1e-11:
            estimate = candidate
            break
        estimate = candidate
    return estimate


def density_eigenvalues(bloch_vector):
    norm = np.linalg.norm(bloch_vector)
    return np.array([
        0.5 * (1.0 - norm),
        0.5 * (1.0 + norm),
    ])


def pure_state_fidelity(estimate, true_bloch):
    return 0.5 * (1.0 + estimate @ true_bloch)


def main():
    polar_angle = 1.1
    azimuth = 0.7
    shots = 256
    particle_count = 1_000_000
    seed = 20261017

    tetrahedron = tetrahedral_vectors()
    state = prepared_state(polar_angle, azimuth)
    true_bloch = state_bloch_vector(state)
    true_probabilities = sic_probabilities(
        true_bloch,
        tetrahedron,
    )

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(shots, true_probabilities)
    frequencies = counts / shots

    linear_estimate = linear_inversion(
        frequencies,
        tetrahedron,
    )
    mle_estimate = maximum_likelihood_estimate(
        frequencies,
        tetrahedron,
    )

    particles = sample_uniform_bloch_ball(
        particle_count,
        rng,
    )
    weights, particle_probabilities = posterior_weights(
        particles,
        counts,
        tetrahedron,
    )
    (
        posterior_mean,
        posterior_covariance,
        intervals,
        radius_interval,
    ) = posterior_statistics(particles, weights)

    posterior_predictive = (
        weights @ particle_probabilities
    )
    predictive_from_mean = sic_probabilities(
        posterior_mean,
        tetrahedron,
    )
    particle_purities = 0.5 * (
        1.0 + np.sum(particles**2, axis=1)
    )
    mean_posterior_purity = float(
        weights @ particle_purities
    )
    posterior_mean_purity = 0.5 * (
        1.0 + posterior_mean @ posterior_mean
    )

    print("Bayesian SIC tomography")
    print(f"Shots:              {shots}")
    print(f"Prior particles:    {particle_count}")
    print(f"Effective particles: {effective_sample_size(weights):.3f}")
    print(
        "Information gain:   "
        f"{information_gain_bits(weights):.6f} bit"
    )
    print(
        "True Bloch vector: "
        + np.array2string(true_bloch, precision=9)
    )
    print(
        "True probabilities: "
        + np.array2string(true_probabilities, precision=9)
    )
    print("Counts:             " + np.array2string(counts))
    print(
        "Observed frequencies: "
        + np.array2string(frequencies, precision=9)
    )

    print("\nPoint estimates")
    print(
        "Linear inversion: "
        + np.array2string(linear_estimate, precision=9)
    )
    print(f"  |r| = {np.linalg.norm(linear_estimate):.9f}")
    print(
        "MLE:              "
        + np.array2string(mle_estimate, precision=9)
    )
    print(f"  |r| = {np.linalg.norm(mle_estimate):.12f}")
    print(
        "Bayesian mean:    "
        + np.array2string(posterior_mean, precision=9)
    )
    print(f"  |r| = {np.linalg.norm(posterior_mean):.9f}")
    print(
        "  eigenvalues = "
        + np.array2string(
            density_eigenvalues(posterior_mean),
            precision=9,
        )
    )
    print(
        "  fidelity to true pure state = "
        f"{pure_state_fidelity(posterior_mean, true_bloch):.9f}"
    )

    print("\nPosterior marginal summaries")
    print("component       5%       median       95%       std")
    for label, interval, variance in zip(
        ("x", "y", "z"),
        intervals,
        np.diag(posterior_covariance),
    ):
        print(
            f"{label:>5s}     {interval[0]:+.6f}  "
            f"{interval[1]:+.6f}  {interval[2]:+.6f}  "
            f"{np.sqrt(variance):.6f}"
        )
    print(
        "Bloch radius 5%/median/95%: "
        + np.array2string(radius_interval, precision=9)
    )
    print("Posterior covariance:")
    print(
        np.array2string(
            posterior_covariance,
            precision=7,
            suppress_small=True,
        )
    )

    print("\nPosterior prediction")
    print(
        "Posterior-predictive SIC probabilities: "
        + np.array2string(posterior_predictive, precision=9)
    )
    print(
        "Probabilities from posterior mean:      "
        + np.array2string(predictive_from_mean, precision=9)
    )
    print(
        "Maximum predictive identity error:      "
        f"{np.max(np.abs(posterior_predictive - predictive_from_mean)):.3e}"
    )
    print(
        "Purity of posterior-mean state: "
        f"{posterior_mean_purity:.9f}"
    )
    print(
        "Posterior mean of purity:       "
        f"{mean_posterior_purity:.9f}"
    )


if __name__ == "__main__":
    main()
