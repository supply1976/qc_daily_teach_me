"""Day 59: Measurement incompatibility in multiparameter metrology."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


def sensing_circuit(theta_x, theta_z):
    """Prepare |+y>, then encode rotations around X and Z."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.s(0)
    circuit.rx(float(theta_x), 0)
    circuit.rz(float(theta_z), 0)
    return circuit


def sensing_state(parameters):
    theta_x, theta_z = parameters
    return Statevector.from_instruction(
        sensing_circuit(theta_x, theta_z)
    )


def pure_state_qfim(parameters, step=1e-6):
    """QFIM and imaginary tangent overlap from finite differences."""
    state = sensing_state(parameters).data
    derivatives = []
    for index in range(2):
        plus = np.array(parameters, dtype=float)
        minus = np.array(parameters, dtype=float)
        plus[index] += step
        minus[index] -= step
        derivatives.append(
            (
                sensing_state(plus).data
                - sensing_state(minus).data
            )
            / (2.0 * step)
        )

    qfim = np.zeros((2, 2), dtype=float)
    curvature = np.zeros((2, 2), dtype=float)
    for row in range(2):
        for column in range(2):
            projected_overlap = (
                np.vdot(derivatives[row], derivatives[column])
                - np.vdot(derivatives[row], state)
                * np.vdot(state, derivatives[column])
            )
            qfim[row, column] = 4.0 * np.real(
                projected_overlap
            )
            curvature[row, column] = 4.0 * np.imag(
                projected_overlap
            )
    return qfim, curvature


def probability_plus(parameters, axis):
    """Probability of the +1 result for a Pauli measurement."""
    state = sensing_state(parameters)
    expectation = float(
        np.real(state.expectation_value(Pauli(axis)))
    )
    return 0.5 * (1.0 + expectation)


def classical_fisher_matrix(parameters, axis, step=1e-6):
    """Classical Fisher matrix for one binary Pauli setting."""
    probability = probability_plus(parameters, axis)
    gradient = np.zeros(2, dtype=float)
    for index in range(2):
        plus = np.array(parameters, dtype=float)
        minus = np.array(parameters, dtype=float)
        plus[index] += step
        minus[index] -= step
        gradient[index] = (
            probability_plus(plus, axis)
            - probability_plus(minus, axis)
        ) / (2.0 * step)

    return np.outer(gradient, gradient) / (
        probability * (1.0 - probability)
    )


def analytic_probabilities(theta_x, theta_z):
    """Bloch-sphere formulas after Rz(theta_z) Rx(theta_x)|+y>."""
    probability_z_plus = 0.5 * (
        1.0 + np.sin(theta_x)
    )
    probability_x_plus = 0.5 * (
        1.0 - np.cos(theta_x) * np.sin(theta_z)
    )
    return probability_x_plus, probability_z_plus


def monte_carlo_estimates(
    true_parameters,
    total_shots,
    repetitions,
    rng,
):
    """Split shots equally between X and Z measurements."""
    theta_x, theta_z = true_parameters
    probability_x, probability_z = analytic_probabilities(
        theta_x,
        theta_z,
    )
    shots_per_axis = total_shots // 2
    x_counts = rng.binomial(
        shots_per_axis,
        probability_x,
        size=repetitions,
    )
    z_counts = rng.binomial(
        shots_per_axis,
        probability_z,
        size=repetitions,
    )
    x_expectation = 2.0 * x_counts / shots_per_axis - 1.0
    z_expectation = 2.0 * z_counts / shots_per_axis - 1.0

    theta_x_estimate = np.arcsin(
        np.clip(z_expectation, -1.0, 1.0)
    )
    theta_z_estimate = np.arcsin(
        np.clip(
            -x_expectation / np.cos(theta_x_estimate),
            -1.0,
            1.0,
        )
    )
    return np.column_stack([
        theta_x_estimate,
        theta_z_estimate,
    ])


def format_matrix(matrix):
    return np.array2string(
        matrix,
        precision=6,
        suppress_small=True,
    )


def main():
    true_parameters = np.array([0.04, -0.03])
    total_shots = 8_192
    repetitions = 4_000
    seed = 20261015

    qfim, curvature = pure_state_qfim(true_parameters)
    analytic_qfim = np.diag([
        1.0,
        np.cos(true_parameters[0]) ** 2,
    ])
    x_fisher = classical_fisher_matrix(
        true_parameters,
        "X",
    )
    z_fisher = classical_fisher_matrix(
        true_parameters,
        "Z",
    )

    # Half of all copies are measured in X, half in Z.
    average_classical_fisher = 0.5 * (
        x_fisher + z_fisher
    )
    quantum_tradeoff_score = float(
        np.trace(
            np.linalg.inv(qfim)
            @ average_classical_fisher
        )
    )

    quantum_covariance_bound = np.linalg.inv(
        total_shots * qfim
    )
    separable_covariance_bound = np.linalg.inv(
        total_shots * average_classical_fisher
    )

    qiskit_probability_errors = []
    analytic_x, analytic_z = analytic_probabilities(
        *true_parameters
    )
    qiskit_probability_errors.append(
        abs(
            probability_plus(true_parameters, "X")
            - analytic_x
        )
    )
    qiskit_probability_errors.append(
        abs(
            probability_plus(true_parameters, "Z")
            - analytic_z
        )
    )

    rng = np.random.default_rng(seed)
    estimates = monte_carlo_estimates(
        true_parameters,
        total_shots,
        repetitions,
        rng,
    )
    observed_covariance = np.cov(estimates, rowvar=False)
    observed_std = np.sqrt(np.diag(observed_covariance))
    observed_bias = np.mean(estimates, axis=0) - true_parameters

    print("Multiparameter measurement incompatibility")
    print(
        "True parameters [theta_x, theta_z]: "
        + np.array2string(true_parameters, precision=6)
    )
    print(f"Total shots:             {total_shots}")
    print(f"Shots per measurement:   {total_shots // 2}")
    print(f"Monte Carlo repetitions: {repetitions}")
    print(
        "Maximum Qiskit-versus-analytic probability error: "
        f"{max(qiskit_probability_errors):.3e}"
    )
    print(
        "Maximum numerical-versus-analytic QFIM error:     "
        f"{np.max(np.abs(qfim - analytic_qfim)):.3e}"
    )

    print("\nQuantum Fisher information matrix:")
    print(format_matrix(qfim))
    print("Imaginary tangent-overlap matrix:")
    print(format_matrix(curvature))
    print(
        "Nonzero off-diagonal imaginary overlap means that "
        "the two locally optimal measurements are incompatible."
    )

    print("\nClassical Fisher matrices:")
    print("X measurement:")
    print(format_matrix(x_fisher))
    print(
        "rank = "
        f"{np.linalg.matrix_rank(x_fisher, tol=1e-10)}"
    )
    print("Z measurement:")
    print(format_matrix(z_fisher))
    print(
        "rank = "
        f"{np.linalg.matrix_rank(z_fisher, tol=1e-10)}"
    )
    print("50/50 measurement mixture, FI per copy:")
    print(format_matrix(average_classical_fisher))
    print(
        "Tr(F_Q^-1 F_C) = "
        f"{quantum_tradeoff_score:.9f}"
    )

    quantum_std = np.sqrt(
        np.diag(quantum_covariance_bound)
    )
    separable_std = np.sqrt(
        np.diag(separable_covariance_bound)
    )
    print("\nUncertainty comparison:")
    print(
        " parameter    naive QFIM std"
        "    separable CRLB std    observed std       bias"
    )
    print("-" * 86)
    names = ["theta_x", "theta_z"]
    for index, name in enumerate(names):
        print(
            f" {name:7s}      {quantum_std[index]:.6e}"
            f"         {separable_std[index]:.6e}"
            f"       {observed_std[index]:.6e}"
            f"   {observed_bias[index]:+.2e}"
        )

    print(
        "\nThe QFIM contains unit information about both directions, "
        "but a single-copy measurement cannot extract both units at once."
    )


if __name__ == "__main__":
    main()
