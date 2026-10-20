"""Day 64: Adaptive Bayesian tomography by expected information gain."""

from math import comb

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


def prepared_state(polar_angle, azimuth):
    """Prepare Rz(azimuth) Ry(polar_angle)|0> with Qiskit."""
    circuit = QuantumCircuit(1)
    circuit.ry(float(polar_angle), 0)
    circuit.rz(float(azimuth), 0)
    return Statevector.from_instruction(circuit)


def state_bloch_vector(state):
    return np.array([
        float(np.real(state.expectation_value(Pauli(axis))))
        for axis in ("X", "Y", "Z")
    ])


def candidate_measurements():
    """Nine experimentally simple projective-measurement axes."""
    root_two = np.sqrt(2.0)
    return [
        ("X", np.array([1.0, 0.0, 0.0])),
        ("Y", np.array([0.0, 1.0, 0.0])),
        ("Z", np.array([0.0, 0.0, 1.0])),
        ("X+Y", np.array([1.0, 1.0, 0.0]) / root_two),
        ("X-Y", np.array([1.0, -1.0, 0.0]) / root_two),
        ("X+Z", np.array([1.0, 0.0, 1.0]) / root_two),
        ("X-Z", np.array([1.0, 0.0, -1.0]) / root_two),
        ("Y+Z", np.array([0.0, 1.0, 1.0]) / root_two),
        ("Y-Z", np.array([0.0, 1.0, -1.0]) / root_two),
    ]


def sample_uniform_bloch_ball(particle_count, rng):
    directions = rng.normal(size=(particle_count, 3))
    directions /= np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )
    radii = rng.random(particle_count) ** (1.0 / 3.0)
    return directions * radii[:, None]


def plus_probabilities(particles, direction):
    probabilities = 0.5 * (1.0 + particles @ direction)
    return np.clip(probabilities, 1e-12, 1.0 - 1e-12)


def batch_information_gain(
    particles,
    weights,
    direction,
    batch_size,
):
    """Mutual information I(Bloch vector; batch outcome) in bits."""
    probabilities = plus_probabilities(particles, direction)
    plus_counts = np.arange(batch_size + 1)
    binomial_coefficients = np.array([
        comb(batch_size, int(count))
        for count in plus_counts
    ])
    likelihoods = (
        binomial_coefficients[None, :]
        * probabilities[:, None] ** plus_counts[None, :]
        * (1.0 - probabilities[:, None])
        ** (batch_size - plus_counts)[None, :]
    )
    predictive = weights @ likelihoods
    ratio = likelihoods / np.clip(
        predictive[None, :],
        1e-300,
        None,
    )
    terms = np.where(
        likelihoods > 0.0,
        likelihoods * np.log2(np.clip(ratio, 1e-300, None)),
        0.0,
    )
    return float(weights @ np.sum(terms, axis=1))


def normalize_log_weights(log_weights):
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def effective_sample_size(weights):
    return float(1.0 / np.sum(weights**2))


def systematic_resample(weights, rng):
    particle_count = weights.size
    positions = (
        rng.random() + np.arange(particle_count)
    ) / particle_count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def data_log_likelihood(
    particles,
    directions,
    plus_counts,
    minus_counts,
):
    if len(directions) == 0:
        return np.zeros(particles.shape[0])
    direction_matrix = np.asarray(directions)
    probabilities = np.clip(
        0.5 * (1.0 + particles @ direction_matrix.T),
        1e-12,
        1.0 - 1e-12,
    )
    return (
        np.log(probabilities) @ np.asarray(plus_counts)
        + np.log1p(-probabilities) @ np.asarray(minus_counts)
    )


def metropolis_rejuvenation(
    particles,
    directions,
    plus_counts,
    minus_counts,
    proposal_scale,
    move_steps,
    rng,
):
    current_log_likelihood = data_log_likelihood(
        particles,
        directions,
        plus_counts,
        minus_counts,
    )
    accepted = 0
    attempted = particles.shape[0] * move_steps

    for _ in range(move_steps):
        proposal = particles + rng.normal(
            scale=proposal_scale,
            size=particles.shape,
        )
        physical = np.linalg.norm(proposal, axis=1) <= 1.0
        proposed_log_likelihood = np.full(
            particles.shape[0],
            -np.inf,
        )
        proposed_log_likelihood[physical] = data_log_likelihood(
            proposal[physical],
            directions,
            plus_counts,
            minus_counts,
        )
        log_ratio = proposed_log_likelihood - current_log_likelihood
        accept = np.log(rng.random(particles.shape[0])) < np.minimum(
            0.0,
            log_ratio,
        )
        particles[accept] = proposal[accept]
        current_log_likelihood[accept] = proposed_log_likelihood[accept]
        accepted += int(np.sum(accept))

    return particles, accepted / attempted


def posterior_statistics(particles, weights):
    mean = weights @ particles
    centered = particles - mean
    covariance = centered.T @ (weights[:, None] * centered)
    return mean, covariance


def pure_state_fidelity(estimate, true_bloch):
    return 0.5 * (1.0 + estimate @ true_bloch)


def run_protocol(
    initial_particles,
    true_bloch,
    candidates,
    batch_size,
    rounds,
    adaptive,
    seed,
):
    """Run either adaptive design or fixed X/Y/Z round-robin design."""
    particle_count = initial_particles.shape[0]
    particles = initial_particles.copy()
    log_weights = np.full(
        particle_count,
        -np.log(particle_count),
    )
    rng = np.random.default_rng(seed)

    measured_directions = []
    observed_plus = []
    observed_minus = []
    history = []

    for round_index in range(rounds):
        weights = normalize_log_weights(log_weights)
        utilities = np.array([
            batch_information_gain(
                particles,
                weights,
                direction,
                batch_size,
            )
            for _, direction in candidates
        ])

        if adaptive:
            choice = int(np.argmax(utilities))
        else:
            choice = round_index % 3

        label, direction = candidates[choice]
        true_plus_probability = 0.5 * (
            1.0 + true_bloch @ direction
        )
        plus_count = int(rng.binomial(
            batch_size,
            true_plus_probability,
        ))
        minus_count = batch_size - plus_count

        probabilities = plus_probabilities(particles, direction)
        log_weights += (
            plus_count * np.log(probabilities)
            + minus_count * np.log1p(-probabilities)
        )
        weights = normalize_log_weights(log_weights)
        ess_before = effective_sample_size(weights)

        measured_directions.append(direction)
        observed_plus.append(plus_count)
        observed_minus.append(minus_count)

        resampled = ess_before < 0.5 * particle_count
        acceptance = None
        if resampled:
            indices = systematic_resample(weights, rng)
            particles = particles[indices].copy()
            log_weights.fill(-np.log(particle_count))
            weights.fill(1.0 / particle_count)
            particles, acceptance = metropolis_rejuvenation(
                particles,
                measured_directions,
                observed_plus,
                observed_minus,
                proposal_scale=0.07,
                move_steps=3,
                rng=rng,
            )

        mean, covariance = posterior_statistics(particles, weights)
        history.append({
            "round": round_index + 1,
            "label": label,
            "plus": plus_count,
            "utility": utilities[choice],
            "ess": ess_before,
            "resampled": resampled,
            "acceptance": acceptance,
            "fidelity": pure_state_fidelity(mean, true_bloch),
            "trace_covariance": float(np.trace(covariance)),
        })

    weights = normalize_log_weights(log_weights)
    mean, covariance = posterior_statistics(particles, weights)
    return {
        "mean": mean,
        "covariance": covariance,
        "weights": weights,
        "history": history,
        "directions": [item["label"] for item in history],
    }


def direction_histogram(direction_labels):
    labels, counts = np.unique(direction_labels, return_counts=True)
    return {
        str(label): int(count)
        for label, count in zip(labels, counts)
    }


def main():
    polar_angle = 1.1
    azimuth = 0.7
    particle_count = 12_000
    batch_size = 8
    rounds = 24
    prior_seed = 20261020

    true_state = prepared_state(polar_angle, azimuth)
    true_bloch = state_bloch_vector(true_state)
    candidates = candidate_measurements()

    prior_rng = np.random.default_rng(prior_seed)
    initial_particles = sample_uniform_bloch_ball(
        particle_count,
        prior_rng,
    )

    adaptive = run_protocol(
        initial_particles,
        true_bloch,
        candidates,
        batch_size,
        rounds,
        adaptive=True,
        seed=prior_seed + 1,
    )
    round_robin = run_protocol(
        initial_particles,
        true_bloch,
        candidates,
        batch_size,
        rounds,
        adaptive=False,
        seed=prior_seed + 2,
    )

    print("Adaptive Bayesian qubit tomography")
    print(f"Particles:        {particle_count}")
    print(f"Batch size:       {batch_size}")
    print(f"Total shots:      {batch_size * rounds}")
    print(
        "True Bloch vector: "
        + np.array2string(true_bloch, precision=9)
    )

    print("\nAdaptive decisions")
    print(
        "round  axis  plus/8  utility(bit)  ESS before  "
        "resample  fidelity  Tr(cov)"
    )
    print("-" * 88)
    for item in adaptive["history"]:
        if item["round"] <= 8 or item["round"] % 4 == 0:
            print(
                f"{item['round']:5d}  "
                f"{item['label']:<4s}  "
                f"{item['plus']:>4d}/8   "
                f"{item['utility']:.6f}   "
                f"{item['ess']:9.2f}   "
                f"{str(item['resampled']):>7s}   "
                f"{item['fidelity']:.6f}  "
                f"{item['trace_covariance']:.6f}"
            )

    print("\nMeasurement allocation")
    print("Adaptive:  " + str(direction_histogram(
        adaptive["directions"],
    )))
    print("Round robin: " + str(direction_histogram(
        round_robin["directions"],
    )))

    print("\nFinal comparison")
    print("protocol       fidelity    |r_mean|    Tr(cov)    final ESS")
    print("-" * 68)
    for name, result in (
        ("adaptive", adaptive),
        ("round robin", round_robin),
    ):
        mean = result["mean"]
        print(
            f"{name:<12s}  "
            f"{pure_state_fidelity(mean, true_bloch):.9f}  "
            f"{np.linalg.norm(mean):.9f}  "
            f"{np.trace(result['covariance']):.9f}  "
            f"{effective_sample_size(result['weights']):9.3f}"
        )
        print(
            "  posterior mean: "
            + np.array2string(mean, precision=9)
        )

    first_weights = np.full(particle_count, 1.0 / particle_count)
    initial_utilities = np.array([
        batch_information_gain(
            initial_particles,
            first_weights,
            direction,
            batch_size,
        )
        for _, direction in candidates
    ])
    print("\nInitial utility range (isotropic prior):")
    print(
        f"min={np.min(initial_utilities):.9f} bit, "
        f"max={np.max(initial_utilities):.9f} bit"
    )


if __name__ == "__main__":
    main()
