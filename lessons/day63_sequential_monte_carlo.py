"""Day 63: Sequential Monte Carlo for Bayesian qubit tomography."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


def tetrahedral_vectors():
    """Four tetrahedral Bloch vectors defining a qubit SIC-POVM."""
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
    """Read the X, Y, and Z expectations of a Qiskit state."""
    return np.array([
        float(np.real(state.expectation_value(Pauli(axis))))
        for axis in ("X", "Y", "Z")
    ])


def sic_probabilities(bloch_vectors, tetrahedron):
    """Return p_k = (1 + r dot n_k) / 4 for one or many states."""
    vectors = np.asarray(bloch_vectors)
    return (1.0 + vectors @ tetrahedron.T) / 4.0


def sample_uniform_bloch_ball(particle_count, rng):
    """Sample a uniform-volume (Hilbert--Schmidt) prior."""
    directions = rng.normal(size=(particle_count, 3))
    directions /= np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )
    radii = rng.random(particle_count) ** (1.0 / 3.0)
    return directions * radii[:, None]


def log_likelihood(particles, counts, tetrahedron):
    probabilities = np.clip(
        sic_probabilities(particles, tetrahedron),
        1e-15,
        None,
    )
    return np.log(probabilities) @ counts


def normalized_weights(log_weights):
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def effective_sample_size(weights):
    return float(1.0 / np.sum(weights**2))


def systematic_resample(weights, rng):
    """Low-variance resampling using one random offset."""
    particle_count = weights.size
    positions = (
        rng.random() + np.arange(particle_count)
    ) / particle_count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def metropolis_rejuvenation(
    particles,
    accumulated_counts,
    tetrahedron,
    proposal_scale,
    move_steps,
    rng,
):
    """Random-walk Metropolis moves that preserve the posterior."""
    if move_steps == 0:
        return particles, 0.0

    current_log_likelihood = log_likelihood(
        particles,
        accumulated_counts,
        tetrahedron,
    )
    accepted = 0
    attempted = particles.shape[0] * move_steps

    for _ in range(move_steps):
        proposal = particles + rng.normal(
            scale=proposal_scale,
            size=particles.shape,
        )
        physical = np.linalg.norm(proposal, axis=1) <= 1.0
        proposal_log_likelihood = np.full(
            particles.shape[0],
            -np.inf,
        )
        proposal_log_likelihood[physical] = log_likelihood(
            proposal[physical],
            accumulated_counts,
            tetrahedron,
        )

        log_acceptance_ratio = (
            proposal_log_likelihood - current_log_likelihood
        )
        accept = np.log(rng.random(particles.shape[0])) < (
            np.minimum(0.0, log_acceptance_ratio)
        )
        particles[accept] = proposal[accept]
        current_log_likelihood[accept] = (
            proposal_log_likelihood[accept]
        )
        accepted += int(np.sum(accept))

    return particles, accepted / attempted


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return np.interp(quantiles, cumulative, values[order])


def posterior_summary(particles, weights):
    mean = weights @ particles
    intervals = np.array([
        weighted_quantile(
            particles[:, axis],
            weights,
            [0.05, 0.50, 0.95],
        )
        for axis in range(3)
    ])
    return mean, intervals


def pure_state_fidelity(estimate, true_bloch):
    return 0.5 * (1.0 + estimate @ true_bloch)


def unique_particle_fraction(particles):
    return np.unique(particles, axis=0).shape[0] / particles.shape[0]


def main():
    polar_angle = 1.1
    azimuth = 0.7
    counts = np.array([134, 58, 35, 29])

    particle_count = 20_000
    batch_size = 16
    resample_threshold = 0.50
    proposal_scale = 0.08
    move_steps = 5
    seed = 20261019

    tetrahedron = tetrahedral_vectors()
    true_state = prepared_state(polar_angle, azimuth)
    true_bloch = state_bloch_vector(true_state)
    true_probabilities = sic_probabilities(
        true_bloch,
        tetrahedron,
    )

    rng = np.random.default_rng(seed)
    initial_particles = sample_uniform_bloch_ball(
        particle_count,
        rng,
    )

    # Expand the fixed counts from Days 61--62 into a shuffled stream.
    outcome_stream = np.repeat(np.arange(4), counts)
    rng.shuffle(outcome_stream)

    smc_particles = initial_particles.copy()
    smc_log_weights = np.full(
        particle_count,
        -np.log(particle_count),
    )
    static_log_weights = smc_log_weights.copy()
    accumulated_counts = np.zeros(4, dtype=int)
    resampling_rounds = []

    print("Sequential Monte Carlo SIC tomography")
    print(f"Measurements:       {np.sum(counts)}")
    print(f"Particles:          {particle_count}")
    print(f"Batch size:         {batch_size}")
    print(f"Resample threshold: {resample_threshold:.2f} N")
    print(f"Metropolis steps:   {move_steps}")
    print("True Bloch vector:  " + np.array2string(
        true_bloch,
        precision=9,
    ))
    print("True SIC probabilities: " + np.array2string(
        true_probabilities,
        precision=9,
    ))
    print("Fixed counts:       " + np.array2string(counts))

    print("\nSequential updates")
    print(
        "round shots   SMC ESS  resample  MH accept  "
        "fidelity  static ESS"
    )
    print("-" * 78)

    for start in range(0, outcome_stream.size, batch_size):
        batch = outcome_stream[start:start + batch_size]
        batch_counts = np.bincount(batch, minlength=4)
        accumulated_counts += batch_counts

        increment = log_likelihood(
            smc_particles,
            batch_counts,
            tetrahedron,
        )
        smc_log_weights += increment
        smc_weights = normalized_weights(smc_log_weights)

        static_log_weights += log_likelihood(
            initial_particles,
            batch_counts,
            tetrahedron,
        )
        static_weights = normalized_weights(static_log_weights)

        smc_ess = effective_sample_size(smc_weights)
        static_ess = effective_sample_size(static_weights)
        resampled = smc_ess < (
            resample_threshold * particle_count
        )
        acceptance = None

        if resampled:
            indices = systematic_resample(smc_weights, rng)
            smc_particles = smc_particles[indices].copy()
            smc_log_weights.fill(-np.log(particle_count))
            smc_weights.fill(1.0 / particle_count)
            smc_particles, acceptance = metropolis_rejuvenation(
                smc_particles,
                accumulated_counts,
                tetrahedron,
                proposal_scale,
                move_steps,
                rng,
            )
            resampling_rounds.append(start // batch_size + 1)

        mean = smc_weights @ smc_particles
        fidelity = pure_state_fidelity(mean, true_bloch)
        acceptance_text = (
            f"{acceptance:7.3f}"
            if acceptance is not None
            else "      -"
        )
        print(
            f"{start // batch_size + 1:5d} "
            f"{start + batch.size:5d} "
            f"{smc_ess:9.2f} "
            f"{str(resampled):>9s} "
            f"{acceptance_text} "
            f"{fidelity:9.6f} "
            f"{static_ess:11.2f}"
        )

    smc_weights = normalized_weights(smc_log_weights)
    static_weights = normalized_weights(static_log_weights)
    smc_mean, smc_intervals = posterior_summary(
        smc_particles,
        smc_weights,
    )
    static_mean, _ = posterior_summary(
        initial_particles,
        static_weights,
    )

    print("\nFinal comparison")
    print("Resampling rounds: " + str(resampling_rounds))
    print(
        "SMC posterior mean:    "
        + np.array2string(smc_mean, precision=9)
    )
    print(f"SMC |r|:              {np.linalg.norm(smc_mean):.9f}")
    print(
        "SMC fidelity:         "
        f"{pure_state_fidelity(smc_mean, true_bloch):.9f}"
    )
    print(f"SMC final ESS:        {effective_sample_size(smc_weights):.3f}")
    print(
        "SMC unique fraction:  "
        f"{unique_particle_fraction(smc_particles):.6f}"
    )
    print(
        "Static posterior mean: "
        + np.array2string(static_mean, precision=9)
    )
    print(
        "Static fidelity:      "
        f"{pure_state_fidelity(static_mean, true_bloch):.9f}"
    )
    print(f"Static final ESS:     {effective_sample_size(static_weights):.3f}")

    print("\nSMC 90% marginal credible intervals")
    print("component       5%       median       95%")
    for label, interval in zip(("x", "y", "z"), smc_intervals):
        print(
            f"{label:>5s}     {interval[0]:+.6f}  "
            f"{interval[1]:+.6f}  {interval[2]:+.6f}"
        )


if __name__ == "__main__":
    main()
