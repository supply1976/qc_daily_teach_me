"""Day 58: Multiparameter Fisher information and identifiability."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def ramsey_circuit(
    angle_error,
    sequence_length,
    control_phase,
):
    """Ramsey circuit for a repeated phase-error probe."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    for _ in range(sequence_length):
        circuit.rz(float(angle_error), 0)
    circuit.rz(float(control_phase), 0)
    circuit.h(0)
    return circuit


def qiskit_probability_zero(
    angle_error,
    sequence_length,
    control_phase,
):
    state = Statevector.from_instruction(
        ramsey_circuit(
            angle_error,
            sequence_length,
            control_phase,
        )
    )
    return float(state.probabilities()[0])


def probability_zero(
    angle_error,
    retention,
    sequence_length,
    control_phase,
):
    contrast = retention**sequence_length
    phase = sequence_length * angle_error + control_phase
    return 0.5 * (1.0 + contrast * np.cos(phase))


def probability_gradient(
    angle_error,
    retention,
    sequence_length,
    control_phase,
):
    """Gradient of p(0) with respect to [angle_error, retention]."""
    phase = sequence_length * angle_error + control_phase
    contrast = retention**sequence_length
    return np.array([
        -0.5
        * sequence_length
        * contrast
        * np.sin(phase),
        0.5
        * sequence_length
        * retention ** (sequence_length - 1)
        * np.cos(phase),
    ])


def fisher_matrix(
    angle_error,
    retention,
    sequence_length,
    control_phase,
):
    """Classical Fisher matrix from one binary measurement."""
    probability = probability_zero(
        angle_error,
        retention,
        sequence_length,
        control_phase,
    )
    gradient = probability_gradient(
        angle_error,
        retention,
        sequence_length,
        control_phase,
    )
    variance = probability * (1.0 - probability)
    return np.outer(gradient, gradient) / variance


def d_optimal_pair(
    angle_error,
    retention,
    sequence_length,
    relative_phases,
):
    """Choose two settings that maximize det(F1 + F2)."""
    candidates = []
    for first in range(len(relative_phases)):
        for second in range(first + 1, len(relative_phases)):
            control_first = (
                relative_phases[first]
                - sequence_length * angle_error
            )
            control_second = (
                relative_phases[second]
                - sequence_length * angle_error
            )
            total = fisher_matrix(
                angle_error,
                retention,
                sequence_length,
                control_first,
            ) + fisher_matrix(
                angle_error,
                retention,
                sequence_length,
                control_second,
            )
            candidates.append({
                "determinant": float(np.linalg.det(total)),
                "indices": (first, second),
                "matrix": total,
            })

    maximum = max(item["determinant"] for item in candidates)
    equivalent = [
        item
        for item in candidates
        if np.isclose(
            item["determinant"],
            maximum,
            rtol=1e-12,
            atol=1e-12,
        )
    ]
    return min(equivalent, key=lambda item: item["indices"])


def monte_carlo_joint_estimation(
    angle_error,
    retention,
    sequence_length,
    shots_per_setting,
    repetitions,
    rng,
):
    """Estimate phase and retention from peak and quadrature data."""
    peak_control = -sequence_length * angle_error
    quadrature_control = (
        np.pi / 2.0 - sequence_length * angle_error
    )
    peak_probability = float(
        probability_zero(
            angle_error,
            retention,
            sequence_length,
            peak_control,
        )
    )
    quadrature_probability = float(
        probability_zero(
            angle_error,
            retention,
            sequence_length,
            quadrature_control,
        )
    )

    peak_counts = rng.binomial(
        shots_per_setting,
        peak_probability,
        size=repetitions,
    )
    quadrature_counts = rng.binomial(
        shots_per_setting,
        quadrature_probability,
        size=repetitions,
    )
    peak_frequency = peak_counts / shots_per_setting
    quadrature_frequency = (
        quadrature_counts / shots_per_setting
    )

    contrast_estimate = np.clip(
        2.0 * peak_frequency - 1.0,
        1e-12,
        1.0,
    )
    retention_estimate = (
        contrast_estimate ** (1.0 / sequence_length)
    )

    # Local linear inversion at quadrature.  The independently estimated
    # contrast is used so the two unknown parameters are inferred jointly.
    angle_estimate = angle_error - (
        2.0 * (quadrature_frequency - 0.5)
        / (sequence_length * contrast_estimate)
    )
    return angle_estimate, retention_estimate


def matrix_text(matrix):
    return np.array2string(
        matrix,
        precision=6,
        suppress_small=True,
    )


def main():
    true_angle_error = 0.073
    true_retention = 0.98
    sequence_length = 25
    shots_per_setting = 4_096
    repetitions = 4_000
    seed = 20261014

    peak_control = -sequence_length * true_angle_error
    quadrature_control = (
        np.pi / 2.0
        - sequence_length * true_angle_error
    )
    peak_fisher = fisher_matrix(
        true_angle_error,
        true_retention,
        sequence_length,
        peak_control,
    )
    quadrature_fisher = fisher_matrix(
        true_angle_error,
        true_retention,
        sequence_length,
        quadrature_control,
    )
    combined_fisher = peak_fisher + quadrature_fisher

    relative_phases = np.linspace(
        0.0,
        2.0 * np.pi,
        16,
        endpoint=False,
    )
    design = d_optimal_pair(
        true_angle_error,
        true_retention,
        sequence_length,
        relative_phases,
    )
    first, second = design["indices"]

    # Qiskit validates the noiseless Ramsey probabilities.  The retention
    # factor is then applied analytically as a Bloch-vector contraction.
    qiskit_errors = []
    for relative_phase in relative_phases:
        control = (
            relative_phase
            - sequence_length * true_angle_error
        )
        qiskit = qiskit_probability_zero(
            true_angle_error,
            sequence_length,
            control,
        )
        analytic = probability_zero(
            true_angle_error,
            retention=1.0,
            sequence_length=sequence_length,
            control_phase=control,
        )
        qiskit_errors.append(abs(qiskit - analytic))

    total_fisher = shots_per_setting * combined_fisher
    predicted_covariance = np.linalg.inv(total_fisher)
    predicted_std = np.sqrt(np.diag(predicted_covariance))

    rng = np.random.default_rng(seed)
    angle_estimates, retention_estimates = (
        monte_carlo_joint_estimation(
            true_angle_error,
            true_retention,
            sequence_length,
            shots_per_setting,
            repetitions,
            rng,
        )
    )
    estimates = np.column_stack([
        angle_estimates,
        retention_estimates,
    ])
    observed_covariance = np.cov(estimates, rowvar=False)
    observed_std = np.sqrt(np.diag(observed_covariance))
    observed_correlation = float(
        observed_covariance[0, 1]
        / (observed_std[0] * observed_std[1])
    )

    print("Multiparameter Fisher information for Ramsey calibration")
    print(
        f"True angle error:       {true_angle_error:+.9f} rad/gate"
    )
    print(f"True retention:         {true_retention:.9f}")
    print(f"Sequence length:        {sequence_length}")
    print(f"Shots per setting:      {shots_per_setting}")
    print(f"Monte Carlo repetitions:{repetitions:9d}")
    print(
        "Maximum Qiskit-versus-analytic probability error: "
        f"{max(qiskit_errors):.3e}"
    )

    print("\nOne-setting Fisher matrices:")
    print("Peak setting (contrast sensitive):")
    print(matrix_text(peak_fisher))
    print(
        "rank = "
        f"{np.linalg.matrix_rank(peak_fisher, tol=1e-10)}"
    )
    print("Quadrature setting (phase sensitive):")
    print(matrix_text(quadrature_fisher))
    print(
        "rank = "
        f"{np.linalg.matrix_rank(quadrature_fisher, tol=1e-10)}"
    )

    print("\nCombined Fisher matrix:")
    print(matrix_text(combined_fisher))
    print(
        "eigenvalues = "
        + np.array2string(
            np.linalg.eigvalsh(combined_fisher),
            precision=6,
        )
    )
    print(
        f"determinant = {np.linalg.det(combined_fisher):.6f}"
    )
    print(
        "condition number = "
        f"{np.linalg.cond(combined_fisher):.6f}"
    )

    print("\nD-optimal pair from 16 candidate relative phases:")
    print(
        "relative phases / pi = "
        f"[{relative_phases[first] / np.pi:.3f}, "
        f"{relative_phases[second] / np.pi:.3f}]"
    )
    print(
        f"maximum determinant = {design['determinant']:.6f}"
    )

    print("\nJoint-estimation uncertainty:")
    print(
        " parameter          predicted std    observed std"
        "       bias"
    )
    print("-" * 68)
    print(
        f" angle error        {predicted_std[0]:.6e}"
        f"      {observed_std[0]:.6e}"
        f"   {np.mean(angle_estimates) - true_angle_error:+.2e}"
    )
    print(
        f" retention          {predicted_std[1]:.6e}"
        f"      {observed_std[1]:.6e}"
        f"   {np.mean(retention_estimates) - true_retention:+.2e}"
    )
    print(
        "Observed parameter correlation: "
        f"{observed_correlation:+.6f}"
    )
    print(
        "A single binary setting supplies only one local data direction; "
        "two complementary phases make both parameters identifiable."
    )


if __name__ == "__main__":
    main()
