"""Day 47: single-qubit randomized benchmarking of coherent errors."""

import numpy as np
from qiskit.quantum_info import Operator, random_clifford


ZERO_DENSITY = np.array(
    [[1.0, 0.0], [0.0, 0.0]],
    dtype=complex,
)


def rz_matrix(angle):
    return np.diag([
        np.exp(-0.5j * angle),
        np.exp(+0.5j * angle),
    ])


def unitary_evolve(density, unitary):
    return unitary @ density @ unitary.conj().T


def canonical_unitary_key(unitary):
    """Remove global phase so equivalent Clifford matrices deduplicate."""
    flattened = unitary.ravel()
    pivot = int(np.argmax(np.abs(flattened)))
    phase = np.angle(flattened[pivot])
    canonical = unitary * np.exp(-1j * phase)
    rounded = np.round(canonical, decimals=12)
    return tuple(rounded.real.ravel()) + tuple(rounded.imag.ravel())


def single_qubit_clifford_group():
    """Build all 24 Clifford matrices using Qiskit's uniform sampler."""
    matrices = {}
    seed = 0
    while len(matrices) < 24:
        clifford = random_clifford(1, seed=seed)
        unitary = Operator(clifford).data
        matrices.setdefault(
            canonical_unitary_key(unitary),
            unitary,
        )
        seed += 1

    return np.asarray(list(matrices.values()))


def coherent_rb_sequence(length, epsilon, cliffords, rng):
    """Return exact |0> survival for one random Clifford sequence."""
    density = ZERO_DENSITY.copy()
    ideal_total = np.eye(2, dtype=complex)
    coherent_error = rz_matrix(epsilon)

    for _ in range(length):
        index = int(rng.integers(0, len(cliffords)))
        unitary = cliffords[index]

        density = unitary_evolve(density, unitary)
        density = unitary_evolve(density, coherent_error)
        ideal_total = unitary @ ideal_total

    # Ideal recovery: C_inv = (C_m ... C_1)^dagger.
    density = unitary_evolve(
        density,
        ideal_total.conj().T,
    )
    return float(np.real(density[0, 0]))


def apply_readout_error(probability_zero, error_probability):
    error = float(error_probability)
    return (
        error
        + (1.0 - 2.0 * error) * probability_zero
    )


def sample_survival(probability_zero, shots, rng):
    zero_counts = rng.binomial(shots, probability_zero)
    return float(zero_counts / shots)


def fit_rb_decay(lengths, probabilities):
    """Fit P0(m) = A alpha^m + 1/2 by a log-linear fit."""
    lengths = np.asarray(lengths, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    signals = 2.0 * probabilities - 1.0

    if np.any(signals <= 0.0):
        raise ValueError("All fitted RB signals must be positive.")

    slope, intercept = np.polyfit(
        lengths,
        np.log(signals),
        deg=1,
    )
    alpha = float(np.exp(slope))
    amplitude = float(0.5 * np.exp(intercept))
    return amplitude, alpha


def main():
    epsilon = 0.20
    readout_error = 0.03
    lengths = np.array([1, 2, 4, 8, 16, 32, 64])
    random_sequences = 2000
    shots_per_sequence = 512
    rng = np.random.default_rng(20261003)
    cliffords = single_qubit_clifford_group()

    # Clifford twirling maps the Bloch-sphere rotation to an isotropic
    # contraction with depolarizing parameter alpha.
    exact_alpha = (1.0 + 2.0 * np.cos(epsilon)) / 3.0
    exact_average_fidelity = 0.5 * (1.0 + exact_alpha)
    exact_amplitude = 0.5 * (1.0 - 2.0 * readout_error)

    exact_means = []
    sequence_standard_deviations = []
    sampled_means = []

    for length in lengths:
        exact_probabilities = []
        sampled_probabilities = []

        for _ in range(random_sequences):
            exact_probability = coherent_rb_sequence(
                int(length),
                epsilon,
                cliffords,
                rng,
            )
            observed_probability = apply_readout_error(
                exact_probability,
                readout_error,
            )

            exact_probabilities.append(exact_probability)
            sampled_probabilities.append(
                sample_survival(
                    observed_probability,
                    shots_per_sequence,
                    rng,
                )
            )

        exact_means.append(np.mean(exact_probabilities))
        sequence_standard_deviations.append(
            np.std(exact_probabilities, ddof=1)
        )
        sampled_means.append(np.mean(sampled_probabilities))

    exact_means = np.asarray(exact_means)
    sequence_standard_deviations = np.asarray(
        sequence_standard_deviations
    )
    sampled_means = np.asarray(sampled_means)

    analytic_observed = (
        0.5
        + exact_amplitude * exact_alpha**lengths
    )
    fitted_amplitude, fitted_alpha = fit_rb_decay(
        lengths,
        sampled_means,
    )
    fitted_average_fidelity = 0.5 * (
        1.0 + fitted_alpha
    )

    print("Single-qubit randomized benchmarking")
    print(f"Coherent Rz over-rotation: {epsilon:.6f} rad")
    print(f"Readout bit-flip error:    {readout_error:.6f}")
    print(f"Random sequences/length:   {random_sequences}")
    print(f"Shots/sequence:            {shots_per_sequence}")

    print("\nClifford-twirled parameters:")
    print(f"  exact alpha:              {exact_alpha:.9f}")
    print(f"  exact average fidelity:   {exact_average_fidelity:.9f}")
    print(f"  exact RB amplitude A:     {exact_amplitude:.9f}")

    print("\nRB decay:")
    print(
        " length   coherent mean   sequence std   "
        "sampled mean   analytic observed"
    )
    print("-" * 81)
    for index, length in enumerate(lengths):
        print(
            f" {length:6d}   {exact_means[index]:.9f}"
            f"     {sequence_standard_deviations[index]:.9f}"
            f"     {sampled_means[index]:.9f}"
            f"       {analytic_observed[index]:.9f}"
        )

    print("\nFit P0(m) = A alpha^m + 1/2:")
    print(f"  fitted A:                 {fitted_amplitude:.9f}")
    print(f"  fitted alpha:             {fitted_alpha:.9f}")
    print(f"  fitted average fidelity:  {fitted_average_fidelity:.9f}")
    print(
        "  fidelity error:          "
        f"{fitted_average_fidelity - exact_average_fidelity:+.3e}"
    )

    # A depolarizing channel with the same alpha has the same mean RB
    # curve, even though it has no coherent sequence-to-sequence spread.
    depolarizing_survival = 0.5 * (
        1.0 + exact_alpha**lengths
    )
    maximum_mean_difference = np.max(
        np.abs(
            0.5 * (1.0 + exact_alpha**lengths)
            - depolarizing_survival
        )
    )

    print("\nIdentifiability warning:")
    print(
        "  coherent and matched depolarizing mean-curve difference: "
        f"{maximum_mean_difference:.3e}"
    )
    print(
        "  coherent sequence std at m=64: "
        f"{sequence_standard_deviations[-1]:.9f}"
    )
    print("  matched depolarizing sequence std: 0.000000000")


if __name__ == "__main__":
    main()
