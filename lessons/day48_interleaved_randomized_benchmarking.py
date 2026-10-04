"""Day 48: interleaved randomized benchmarking of a target X gate."""

import numpy as np
from qiskit.quantum_info import Operator, random_clifford


I2 = np.eye(2, dtype=complex)
X_GATE = np.array(
    [[0.0, 1.0], [1.0, 0.0]],
    dtype=complex,
)
ZERO_DENSITY = np.array(
    [[1.0, 0.0], [0.0, 0.0]],
    dtype=complex,
)


def unitary_evolve(density, unitary):
    return unitary @ density @ unitary.conj().T


def depolarizing_channel(density, alpha):
    """Shrink the Bloch vector by alpha."""
    return alpha * density + (1.0 - alpha) * I2 / 2.0


def canonical_unitary_key(unitary):
    flattened = unitary.ravel()
    pivot = int(np.argmax(np.abs(flattened)))
    phase = np.angle(flattened[pivot])
    canonical = unitary * np.exp(-1j * phase)
    rounded = np.round(canonical, decimals=12)
    return tuple(rounded.real.ravel()) + tuple(rounded.imag.ravel())


def single_qubit_clifford_group():
    """Collect the 24 single-qubit Clifford matrices with Qiskit."""
    matrices = {}
    seed = 0

    while len(matrices) < 24:
        unitary = Operator(
            random_clifford(1, seed=seed)
        ).data
        matrices.setdefault(
            canonical_unitary_key(unitary),
            unitary,
        )
        seed += 1

    return np.asarray(list(matrices.values()))


def rb_survival_probability(
    clifford_indices,
    cliffords,
    reference_alpha,
    target_alpha=None,
):
    """Simulate a reference or X-interleaved RB sequence exactly."""
    density = ZERO_DENSITY.copy()
    ideal_total = I2.copy()

    for index in clifford_indices:
        clifford = cliffords[index]

        density = unitary_evolve(density, clifford)
        density = depolarizing_channel(
            density,
            reference_alpha,
        )
        ideal_total = clifford @ ideal_total

        if target_alpha is not None:
            density = unitary_evolve(density, X_GATE)
            density = depolarizing_channel(
                density,
                target_alpha,
            )
            ideal_total = X_GATE @ ideal_total

    # The recovery is ideal in this teaching example. A fixed recovery
    # error would be absorbed mainly into the fitted SPAM amplitude.
    recovery = ideal_total.conj().T
    density = unitary_evolve(density, recovery)
    return float(np.real(density[0, 0]))


def apply_readout_error(probability_zero, error_probability):
    error = float(error_probability)
    return (
        error
        + (1.0 - 2.0 * error) * probability_zero
    )


def sample_probability(probability_zero, shots, rng):
    zero_counts = rng.binomial(shots, probability_zero)
    return float(zero_counts / shots)


def fit_decay(lengths, probabilities):
    """Fit P0(m) = A alpha^m + 1/2."""
    lengths = np.asarray(lengths, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    signal = 2.0 * probabilities - 1.0

    if np.any(signal <= 0.0):
        raise ValueError("RB signal must remain positive for log fitting.")

    slope, intercept = np.polyfit(
        lengths,
        np.log(signal),
        deg=1,
    )
    return (
        float(0.5 * np.exp(intercept)),
        float(np.exp(slope)),
    )


def main():
    reference_alpha = 0.995
    target_alpha = 0.980
    readout_error = 0.04
    lengths = np.array([1, 2, 4, 8, 16, 32, 64])
    random_sequences = 500
    shots_per_sequence = 512
    rng = np.random.default_rng(20261004)
    cliffords = single_qubit_clifford_group()

    reference_means = []
    interleaved_means = []

    for length in lengths:
        reference_samples = []
        interleaved_samples = []

        for _ in range(random_sequences):
            indices = rng.integers(
                0,
                len(cliffords),
                size=int(length),
            )

            reference_probability = rb_survival_probability(
                indices,
                cliffords,
                reference_alpha,
            )
            interleaved_probability = rb_survival_probability(
                indices,
                cliffords,
                reference_alpha,
                target_alpha,
            )

            reference_probability = apply_readout_error(
                reference_probability,
                readout_error,
            )
            interleaved_probability = apply_readout_error(
                interleaved_probability,
                readout_error,
            )

            reference_samples.append(
                sample_probability(
                    reference_probability,
                    shots_per_sequence,
                    rng,
                )
            )
            interleaved_samples.append(
                sample_probability(
                    interleaved_probability,
                    shots_per_sequence,
                    rng,
                )
            )

        reference_means.append(np.mean(reference_samples))
        interleaved_means.append(np.mean(interleaved_samples))

    reference_means = np.asarray(reference_means)
    interleaved_means = np.asarray(interleaved_means)

    fitted_reference_amplitude, fitted_reference_alpha = fit_decay(
        lengths,
        reference_means,
    )
    fitted_interleaved_amplitude, fitted_interleaved_alpha = fit_decay(
        lengths,
        interleaved_means,
    )

    fitted_target_alpha = (
        fitted_interleaved_alpha / fitted_reference_alpha
    )
    fitted_target_fidelity = 0.5 * (
        1.0 + fitted_target_alpha
    )
    exact_target_fidelity = 0.5 * (1.0 + target_alpha)
    naive_interleaved_fidelity = 0.5 * (
        1.0 + fitted_interleaved_alpha
    )

    amplitude = 0.5 * (1.0 - 2.0 * readout_error)
    analytic_reference = (
        0.5 + amplitude * reference_alpha**lengths
    )
    analytic_interleaved = (
        0.5
        + amplitude
        * (reference_alpha * target_alpha) ** lengths
    )

    print("Single-qubit interleaved randomized benchmarking")
    print(f"Reference alpha:             {reference_alpha:.9f}")
    print(f"Target-X alpha:              {target_alpha:.9f}")
    print(
        "Exact interleaved alpha:   "
        f"{reference_alpha * target_alpha:.9f}"
    )
    print(f"Readout bit-flip error:      {readout_error:.6f}")
    print(f"Random sequences/length:     {random_sequences}")
    print(f"Shots/sequence/circuit:      {shots_per_sequence}")

    print("\nSurvival probabilities:")
    print(
        " length   reference sampled   reference exact   "
        "interleaved sampled   interleaved exact"
    )
    print("-" * 91)
    for index, length in enumerate(lengths):
        print(
            f" {length:6d}      {reference_means[index]:.9f}"
            f"        {analytic_reference[index]:.9f}"
            f"           {interleaved_means[index]:.9f}"
            f"          {analytic_interleaved[index]:.9f}"
        )

    print("\nFitted decay parameters:")
    print(
        f"  reference:   A={fitted_reference_amplitude:.9f}, "
        f"alpha={fitted_reference_alpha:.9f}"
    )
    print(
        f"  interleaved: A={fitted_interleaved_amplitude:.9f}, "
        f"alpha={fitted_interleaved_alpha:.9f}"
    )

    print("\nTarget-gate estimate:")
    print(f"  alpha_int / alpha_ref:     {fitted_target_alpha:.9f}")
    print(f"  estimated X fidelity:      {fitted_target_fidelity:.9f}")
    print(f"  exact X fidelity:          {exact_target_fidelity:.9f}")
    print(
        "  fidelity estimation error: "
        f"{fitted_target_fidelity - exact_target_fidelity:+.3e}"
    )
    print(
        "  naive fidelity without reference correction: "
        f"{naive_interleaved_fidelity:.9f}"
    )


if __name__ == "__main__":
    main()
