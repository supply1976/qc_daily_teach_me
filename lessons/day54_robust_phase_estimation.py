"""Day 54: Multiscale robust phase estimation and phase unwrapping."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


def phase_probe_circuit(gate_angle, sequence_length):
    """Prepare |+>, repeat Rz, and return the probe circuit."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    for _ in range(sequence_length):
        circuit.rz(float(gate_angle), 0)
    return circuit


def qiskit_xy_expectations(gate_angle, sequence_length):
    """Compute exact X/Y expectations of the phase probe."""
    state = Statevector.from_instruction(
        phase_probe_circuit(gate_angle, sequence_length)
    )
    x_expectation = float(
        np.real(state.expectation_value(Pauli("X")))
    )
    y_expectation = float(
        np.real(state.expectation_value(Pauli("Y")))
    )
    return x_expectation, y_expectation


def wrap_angle(angle):
    """Map angles to (-pi, pi]."""
    return np.angle(np.exp(1.0j * angle))


def observed_expectations(
    gate_angle,
    sequence_length,
    depolarizing_alpha,
    x_offset,
    y_offset,
):
    """Add contrast decay and small bounded SPAM offsets."""
    x_ideal, y_ideal = qiskit_xy_expectations(
        gate_angle,
        sequence_length,
    )
    contrast = depolarizing_alpha**sequence_length
    x_observed = np.clip(
        contrast * x_ideal + x_offset,
        -1.0,
        1.0,
    )
    y_observed = np.clip(
        contrast * y_ideal + y_offset,
        -1.0,
        1.0,
    )
    return x_observed, y_observed


def sample_expectation(expectation, shots, repetitions, rng):
    """Sample a Pauli expectation using binomial shot statistics."""
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
    return (2.0 * plus_counts - shots) / shots


def multiscale_phase_estimation(
    ideal_angle,
    actual_angle,
    sequence_lengths,
    depolarizing_alpha,
    x_offset,
    y_offset,
    shots,
    repetitions,
    rng,
):
    """Estimate a gate-angle error by recursively unwrapping phases."""
    estimates = None
    history = []

    for sequence_length in sequence_lengths:
        x_exact, y_exact = observed_expectations(
            actual_angle,
            sequence_length,
            depolarizing_alpha,
            x_offset,
            y_offset,
        )
        x_samples = sample_expectation(
            x_exact,
            shots,
            repetitions,
            rng,
        )
        y_samples = sample_expectation(
            y_exact,
            shots,
            repetitions,
            rng,
        )

        measured_phase = np.arctan2(y_samples, x_samples)
        wrapped_residual = wrap_angle(
            measured_phase - sequence_length * ideal_angle
        )
        naive_estimates = wrapped_residual / sequence_length

        if estimates is None:
            estimates = naive_estimates
        else:
            # Choose the 2*pi branch closest to the previous estimate.
            winding_number = np.rint(
                (
                    sequence_length * estimates
                    - wrapped_residual
                )
                / (2.0 * np.pi)
            )
            estimates = (
                wrapped_residual
                + 2.0 * np.pi * winding_number
            ) / sequence_length

        true_error = actual_angle - ideal_angle
        branch_tolerance = np.pi / sequence_length
        branch_success = np.mean(
            np.abs(estimates - true_error) < branch_tolerance
        )
        history.append({
            "length": sequence_length,
            "contrast": depolarizing_alpha**sequence_length,
            "true_wrapped": float(
                wrap_angle(sequence_length * true_error)
            ),
            "naive_mean": float(np.mean(naive_estimates)),
            "rpe_mean": float(np.mean(estimates)),
            "rpe_std": float(np.std(estimates, ddof=1)),
            "success": float(branch_success),
        })

    return history, estimates


def main():
    ideal_angle = np.pi / 2
    angle_error = 0.080
    actual_angle = ideal_angle + angle_error
    sequence_lengths = [1, 2, 4, 8, 16, 32, 64, 128]
    depolarizing_alpha = 0.995
    x_offset = 0.015
    y_offset = -0.012
    shots = 4096
    repetitions = 2000
    rng = np.random.default_rng(20261010)

    # Check the Qiskit circuits against the analytic phase formula.
    qiskit_errors = []
    for length in sequence_lengths:
        x_qiskit, y_qiskit = qiskit_xy_expectations(
            actual_angle,
            length,
        )
        qiskit_errors.extend([
            abs(x_qiskit - np.cos(length * actual_angle)),
            abs(y_qiskit - np.sin(length * actual_angle)),
        ])

    history, final_estimates = multiscale_phase_estimation(
        ideal_angle,
        actual_angle,
        sequence_lengths,
        depolarizing_alpha,
        x_offset,
        y_offset,
        shots,
        repetitions,
        rng,
    )

    final = history[-1]
    print("Robust phase estimation with multiscale unwrapping")
    print(f"Ideal Rz angle:          {ideal_angle:.9f} rad")
    print(f"True angle error:        {angle_error:.9f} rad/gate")
    print(f"Per-gate Bloch shrink:   {depolarizing_alpha:.6f}")
    print(f"SPAM offsets (X, Y):     ({x_offset:+.3f}, {y_offset:+.3f})")
    print(f"Shots per X/Y setting:   {shots}")
    print(f"Monte Carlo repetitions: {repetitions}")
    print(
        "Maximum Qiskit-versus-analytic error: "
        f"{max(qiskit_errors):.3e}"
    )

    print("\nMultiscale estimates:")
    print(
        "   L   contrast   wrapped L*error"
        "   naive error    RPE error      RPE std    success"
    )
    print("-" * 94)
    for result in history:
        print(
            f"{result['length']:4d}   {result['contrast']:.6f}"
            f"     {result['true_wrapped']:+.6f}"
            f"      {result['naive_mean']:+.6f}"
            f"      {result['rpe_mean']:+.6f}"
            f"    {result['rpe_std']:.3e}"
            f"    {100.0 * result['success']:6.2f}%"
        )

    final_bias = float(np.mean(final_estimates) - angle_error)
    print("\nFinal comparison at the longest sequence:")
    print(f"Naive wrapped estimate:  {final['naive_mean']:+.9f} rad/gate")
    print(f"Multiscale RPE estimate: {final['rpe_mean']:+.9f} rad/gate")
    print(f"RPE bias:                {final_bias:+.3e} rad/gate")
    print(f"RPE standard deviation:  {final['rpe_std']:.3e} rad/gate")
    print(
        "The short sequences locate the correct 2*pi branch; "
        "the long sequences supply fine precision."
    )


if __name__ == "__main__":
    main()
