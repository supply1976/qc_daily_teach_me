"""Day 53: Long-sequence GST and coherent-error amplification."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


def repeated_rx_state(angle, length):
    """Prepare Rx(angle)^length |0> with an explicit Qiskit circuit."""
    circuit = QuantumCircuit(1)
    for _ in range(length):
        circuit.rx(float(angle), 0)
    return circuit, Statevector.from_instruction(circuit)


def qiskit_bloch_yz(angle, length):
    """Return the ideal Y and Z expectations from the Qiskit state."""
    _, state = repeated_rx_state(angle, length)
    y_expectation = float(
        np.real(state.expectation_value(Pauli("Y")))
    )
    z_expectation = float(
        np.real(state.expectation_value(Pauli("Z")))
    )
    return y_expectation, z_expectation


def noisy_bloch_yz(angle, length, depolarizing_alpha):
    """Apply a simple per-gate depolarizing shrinkage to the Bloch vector."""
    qiskit_y, qiskit_z = qiskit_bloch_yz(angle, length)
    contrast = depolarizing_alpha**length
    return contrast * qiskit_y, contrast * qiskit_z


def sample_expectations(expectations, shots, repetitions, rng):
    """Sample Pauli expectations from independent finite-shot experiments."""
    estimates = []
    for expectation in expectations:
        probability_plus = np.clip(
            0.5 * (1.0 + expectation),
            0.0,
            1.0,
        )
        plus_counts = rng.binomial(
            shots,
            probability_plus,
            size=repetitions,
        )
        estimates.append((2.0 * plus_counts - shots) / shots)
    return estimates


def wrap_angle(angle):
    """Map an angle to (-pi, pi]."""
    return np.angle(np.exp(1.0j * angle))


def estimate_overrotation(
    sequence_length,
    ideal_angle,
    actual_angle,
    depolarizing_alpha,
    shots,
    repetitions,
    rng,
):
    """Estimate the small angle error from Y/Z measurements."""
    y_exact, z_exact = noisy_bloch_yz(
        actual_angle,
        sequence_length,
        depolarizing_alpha,
    )
    y_samples, z_samples = sample_expectations(
        (y_exact, z_exact),
        shots,
        repetitions,
        rng,
    )

    # Rx(phi)|0> has <Y>=-sin(phi), <Z>=cos(phi).
    measured_phase = np.arctan2(-y_samples, z_samples)
    amplified_error = wrap_angle(
        measured_phase - sequence_length * ideal_angle
    )
    estimates = amplified_error / sequence_length

    # Delta-method prediction for the phase uncertainty.
    variance_y = (1.0 - y_exact**2) / shots
    variance_z = (1.0 - z_exact**2) / shots
    radius_squared = y_exact**2 + z_exact**2
    phase_variance = (
        z_exact**2 * variance_y
        + y_exact**2 * variance_z
    ) / radius_squared**2

    return {
        "contrast": depolarizing_alpha**sequence_length,
        "amplified_error": wrap_angle(
            sequence_length * (actual_angle - ideal_angle)
        ),
        "mean": float(np.mean(estimates)),
        "bias": float(
            np.mean(estimates) - (actual_angle - ideal_angle)
        ),
        "std": float(np.std(estimates, ddof=1)),
        "predicted_std": float(
            np.sqrt(phase_variance) / sequence_length
        ),
    }


def main():
    ideal_angle = np.pi / 2
    overrotation = 0.008
    actual_angle = ideal_angle + overrotation
    depolarizing_alpha = 0.99
    sequence_lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    shots = 2048
    repetitions = 2000
    rng = np.random.default_rng(20261009)

    # Confirm that Qiskit's circuit agrees with the analytic rotation formula.
    qiskit_errors = []
    for length in sequence_lengths:
        qiskit_y, qiskit_z = qiskit_bloch_yz(
            actual_angle,
            length,
        )
        analytic_y = -np.sin(length * actual_angle)
        analytic_z = np.cos(length * actual_angle)
        qiskit_errors.extend([
            abs(qiskit_y - analytic_y),
            abs(qiskit_z - analytic_z),
        ])

    results = []
    for length in sequence_lengths:
        result = estimate_overrotation(
            length,
            ideal_angle,
            actual_angle,
            depolarizing_alpha,
            shots,
            repetitions,
            rng,
        )
        results.append((length, result))

    observed_best = min(results, key=lambda item: item[1]["std"])
    predicted_best = min(
        results,
        key=lambda item: item[1]["predicted_std"],
    )

    print("Long-sequence GST: coherent-error amplification")
    print(f"Ideal gate angle:       pi/2 = {ideal_angle:.9f} rad")
    print(f"True over-rotation:     {overrotation:.9f} rad/gate")
    print(f"Per-gate Bloch shrink:  {depolarizing_alpha:.5f}")
    print(f"Shots per Y/Z setting:  {shots}")
    print(f"Monte Carlo repetitions:{repetitions}")
    print(
        "Maximum Qiskit-versus-analytic Bloch error: "
        f"{max(qiskit_errors):.3e}"
    )

    print("\nError-amplification experiment:")
    print(
        "   L   contrast    L*epsilon    mean epsilon"
        "       bias       observed std    predicted std"
    )
    print("-" * 100)
    for length, result in results:
        print(
            f"{length:4d}   {result['contrast']:.6f}"
            f"   {result['amplified_error']:+.6f}"
            f"    {result['mean']:+.9f}"
            f"   {result['bias']:+.2e}"
            f"     {result['std']:.3e}"
            f"        {result['predicted_std']:.3e}"
        )

    first_std = results[0][1]["std"]
    best_length, best_result = observed_best
    print("\nPrecision summary:")
    print(
        f"Observed best length:   L={best_length}, "
        f"std={best_result['std']:.3e} rad/gate"
    )
    print(
        f"Predicted best length:  L={predicted_best[0]}, "
        f"std={predicted_best[1]['predicted_std']:.3e} rad/gate"
    )
    print(
        "Precision improvement over L=1: "
        f"{first_std / best_result['std']:.2f}x"
    )
    print(
        "The longest sequence is not always best: coherent phase grows "
        "as L, while depolarizing contrast decays as alpha^L."
    )


if __name__ == "__main__":
    main()
