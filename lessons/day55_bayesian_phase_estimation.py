"""Day 55: Bayesian phase estimation with a full posterior distribution."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


def phase_probe_circuit(ideal_angle, actual_angle, sequence_length):
    """Prepare |+>, apply the actual gate, then remove ideal evolution."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    for _ in range(sequence_length):
        circuit.rz(float(actual_angle), 0)
    circuit.rz(float(-sequence_length * ideal_angle), 0)
    return circuit


def qiskit_xy_expectations(
    ideal_angle,
    actual_angle,
    sequence_length,
):
    state = Statevector.from_instruction(
        phase_probe_circuit(
            ideal_angle,
            actual_angle,
            sequence_length,
        )
    )
    x_expectation = float(
        np.real(state.expectation_value(Pauli("X")))
    )
    y_expectation = float(
        np.real(state.expectation_value(Pauli("Y")))
    )
    return x_expectation, y_expectation


def sample_counts(expectation, shots, rng):
    probability_plus = np.clip(
        0.5 * (1.0 + expectation),
        0.0,
        1.0,
    )
    plus = int(rng.binomial(shots, probability_plus))
    return plus, shots - plus


def log_binomial_likelihood(plus, minus, probability_plus):
    """Parameter-dependent part of the binomial log likelihood."""
    probability_plus = np.clip(
        probability_plus,
        1e-12,
        1.0 - 1e-12,
    )
    return (
        plus * np.log(probability_plus)
        + minus * np.log1p(-probability_plus)
    )


def bayesian_update(
    posterior,
    error_grid,
    sequence_length,
    contrast,
    x_counts,
    y_counts,
):
    phase_grid = sequence_length * error_grid
    probability_x_plus = 0.5 * (
        1.0 + contrast * np.cos(phase_grid)
    )
    probability_y_plus = 0.5 * (
        1.0 + contrast * np.sin(phase_grid)
    )

    log_likelihood = log_binomial_likelihood(
        *x_counts,
        probability_x_plus,
    ) + log_binomial_likelihood(
        *y_counts,
        probability_y_plus,
    )

    log_posterior = np.log(
        np.clip(posterior, 1e-300, None)
    ) + log_likelihood
    log_posterior -= np.max(log_posterior)
    updated = np.exp(log_posterior)
    return updated / np.sum(updated)


def posterior_summary(error_grid, posterior):
    mean = float(np.sum(error_grid * posterior))
    variance = float(
        np.sum((error_grid - mean) ** 2 * posterior)
    )
    cdf = np.cumsum(posterior)
    lower = float(error_grid[np.searchsorted(cdf, 0.025)])
    upper = float(error_grid[np.searchsorted(cdf, 0.975)])
    maximum = float(error_grid[np.argmax(posterior)])
    entropy = float(
        -np.sum(
            posterior
            * np.log2(np.clip(posterior, 1e-300, None))
        )
    )
    information_gain = np.log2(posterior.size) - entropy
    return {
        "mean": mean,
        "std": np.sqrt(variance),
        "map": maximum,
        "lower": lower,
        "upper": upper,
        "information_gain": information_gain,
    }


def posterior_modes(error_grid, posterior, relative_height=0.05):
    interior = np.flatnonzero(
        (posterior[1:-1] > posterior[:-2])
        & (posterior[1:-1] > posterior[2:])
        & (
            posterior[1:-1]
            > relative_height * np.max(posterior)
        )
    ) + 1
    order = interior[np.argsort(posterior[interior])[::-1]]
    return error_grid[order]


def main():
    ideal_angle = np.pi / 2
    true_error = 0.073
    actual_angle = ideal_angle + true_error
    sequence_lengths = [1, 2, 4, 8, 16, 32, 64]
    depolarizing_alpha = 0.995
    shots = 128
    error_grid = np.linspace(-0.15, 0.15, 12001)
    rng = np.random.default_rng(20261011)

    posterior = np.ones_like(error_grid)
    posterior /= np.sum(posterior)
    prior = posterior.copy()
    records = []
    qiskit_errors = []

    for length in sequence_lengths:
        x_ideal, y_ideal = qiskit_xy_expectations(
            ideal_angle,
            actual_angle,
            length,
        )
        qiskit_errors.extend([
            abs(x_ideal - np.cos(length * true_error)),
            abs(y_ideal - np.sin(length * true_error)),
        ])

        contrast = depolarizing_alpha**length
        x_counts = sample_counts(
            contrast * x_ideal,
            shots,
            rng,
        )
        y_counts = sample_counts(
            contrast * y_ideal,
            shots,
            rng,
        )
        posterior = bayesian_update(
            posterior,
            error_grid,
            length,
            contrast,
            x_counts,
            y_counts,
        )
        records.append({
            "length": length,
            "contrast": contrast,
            "x_plus": x_counts[0],
            "y_plus": y_counts[0],
            **posterior_summary(error_grid, posterior),
        })

    # Reuse the longest-sequence data, but discard all shorter experiments.
    longest = records[-1]
    long_only = bayesian_update(
        prior,
        error_grid,
        longest["length"],
        longest["contrast"],
        (longest["x_plus"], shots - longest["x_plus"]),
        (longest["y_plus"], shots - longest["y_plus"]),
    )
    long_only_summary = posterior_summary(
        error_grid,
        long_only,
    )
    long_only_modes = posterior_modes(error_grid, long_only)
    multiscale_modes = posterior_modes(error_grid, posterior)

    print("Bayesian quantum phase estimation")
    print(f"True gate-angle error:   {true_error:+.9f} rad/gate")
    print("Prior interval:          [-0.150, +0.150] rad/gate")
    print(f"Shots per X/Y basis:     {shots}")
    print(f"Per-gate Bloch shrink:   {depolarizing_alpha:.6f}")
    print(
        "Maximum Qiskit-versus-analytic error: "
        f"{max(qiskit_errors):.3e}"
    )

    print("\nSequential posterior updates:")
    print(
        "   L   contrast   X+   Y+      MAP error"
        "     posterior std     95% width    info(bits)"
    )
    print("-" * 97)
    for result in records:
        width = result["upper"] - result["lower"]
        print(
            f"{result['length']:4d}   {result['contrast']:.6f}"
            f"  {result['x_plus']:3d}  {result['y_plus']:3d}"
            f"    {result['map']:+.7f}"
            f"      {result['std']:.3e}"
            f"       {width:.3e}"
            f"      {result['information_gain']:.3f}"
        )

    final = records[-1]
    print("\nFinal multiscale posterior:")
    print(f"posterior mean:          {final['mean']:+.9f}")
    print(f"MAP estimate:            {final['map']:+.9f}")
    print(f"posterior std:           {final['std']:.3e}")
    print(
        "95% credible interval:  "
        f"[{final['lower']:+.9f}, {final['upper']:+.9f}]"
    )
    print(
        "resolved modes:         "
        f"{np.array2string(multiscale_modes, precision=6)}"
    )

    print("\nUsing only the L=64 experiment:")
    print(f"posterior mean:          {long_only_summary['mean']:+.9f}")
    print(f"posterior std:           {long_only_summary['std']:.3e}")
    print(
        "significant modes:      "
        f"{np.array2string(long_only_modes, precision=6)}"
    )
    print(
        "Long sequences make narrow likelihood peaks; short sequences "
        "tell the posterior which periodic peak is physically correct."
    )


if __name__ == "__main__":
    main()
