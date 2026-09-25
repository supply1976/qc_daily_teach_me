"""Day 39: train a non-commuting transverse-field quantum Boltzmann machine."""

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, state_fidelity


BETA = 1.0


def hamiltonian(parameters):
    """H(J, Gamma) = -J ZZ - Gamma (X0 + X1)."""
    coupling, transverse_field = parameters
    return SparsePauliOp.from_list([
        ("ZZ", -float(coupling)),
        ("IX", -float(transverse_field)),
        ("XI", -float(transverse_field)),
    ]).to_matrix()


def gibbs_state(parameters, beta=BETA):
    """Return exp(-beta H) / Z using a stable eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian(parameters))
    weights = np.exp(-beta * (eigenvalues - eigenvalues[0]))
    density = (
        eigenvectors
        @ np.diag(weights / np.sum(weights))
        @ eigenvectors.conj().T
    )
    return DensityMatrix(density)


def expectation(density, label):
    operator = SparsePauliOp.from_list([(label, 1.0)])
    return float(np.real(density.expectation_value(operator)))


def sufficient_statistics(density):
    """Observables conjugate to J and Gamma in the Hamiltonian."""
    return np.array([
        expectation(density, "ZZ"),
        expectation(density, "IX") + expectation(density, "XI"),
    ])


def matrix_log_positive(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-15, None)
    return (
        eigenvectors
        @ np.diag(np.log(eigenvalues))
        @ eigenvectors.conj().T
    )


def quantum_relative_entropy(target, model):
    value = np.trace(
        target.data
        @ (
            matrix_log_positive(target.data)
            - matrix_log_positive(model.data)
        )
    )
    return float(np.real(value))


def loss(parameters, target, beta=BETA):
    return quantum_relative_entropy(target, gibbs_state(parameters, beta))


def analytic_gradient(parameters, target, beta=BETA):
    """Gradient of D(target || rho_parameters)."""
    model = gibbs_state(parameters, beta)
    return beta * (
        sufficient_statistics(model)
        - sufficient_statistics(target)
    )


def finite_difference_gradient(parameters, target, epsilon=1e-6):
    gradient = np.zeros_like(parameters)
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (
            loss(plus, target) - loss(minus, target)
        ) / (2 * epsilon)
    return gradient


def train(target, initial, steps=200, learning_rate=0.08):
    parameters = np.array(initial, dtype=float)
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    history = {}

    for step in range(steps + 1):
        if step in {0, 1, 10, 25, 50, 100, 200}:
            history[step] = (
                parameters.copy(),
                loss(parameters, target),
            )
        if step == steps:
            break

        gradient = analytic_gradient(parameters, target)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1 - 0.9 ** (step + 1))
        corrected_second = second_moment / (1 - 0.999 ** (step + 1))
        parameters -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )

    return parameters, history


def diagonal_probabilities(density):
    return np.real(np.diag(density.data))


def main():
    target_parameters = np.array([0.9, 0.65])
    target = gibbs_state(target_parameters)
    target_statistics = sufficient_statistics(target)

    print("Target parameters:")
    print(f"  J* = {target_parameters[0]:.9f}")
    print(f"  Gamma* = {target_parameters[1]:.9f}")
    print("Target sufficient statistics:")
    print(f"  <ZZ> = {target_statistics[0]:+.9f}")
    print(f"  average <X> = {0.5 * target_statistics[1]:+.9f}\n")

    test_parameters = np.array([-0.35, 0.20])
    analytic = analytic_gradient(test_parameters, target)
    numerical = finite_difference_gradient(test_parameters, target)
    print("Gradient check at [J, Gamma] = [-0.35, 0.20]:")
    print("  analytic:        ", np.array2string(analytic, precision=9))
    print("  finite difference:", np.array2string(numerical, precision=9))
    print(f"  maximum error:    {np.max(np.abs(analytic - numerical)):.3e}\n")

    learned, history = train(
        target,
        initial=np.array([-0.35, 0.20]),
    )
    print("Training with <ZZ> and <X0+X1>:")
    print(" step     relative entropy          J          Gamma")
    print("-" * 58)
    for step, (parameters, relative_entropy) in history.items():
        print(
            f"{step:5d}     {relative_entropy:.12e}"
            f"   {parameters[0]:+.6f}   {parameters[1]:+.6f}"
        )

    learned_state = gibbs_state(learned)
    learned_statistics = sufficient_statistics(learned_state)
    print("\nFinal comparison:")
    print(f"  learned [J, Gamma] = {np.array2string(learned, precision=9)}")
    print(f"  learned <ZZ>       = {learned_statistics[0]:+.9f}")
    print(f"  learned average<X> = {0.5 * learned_statistics[1]:+.9f}")
    print(f"  state fidelity     = {state_fidelity(target, learned_state):.12f}")

    target_probabilities = diagonal_probabilities(target)
    diagonal_coupling = 0.5 * np.log(
        target_probabilities[0] / target_probabilities[1]
    )
    diagonal_model = gibbs_state(np.array([diagonal_coupling, 0.0]))
    diagonal_model_probabilities = diagonal_probabilities(diagonal_model)
    total_variation = 0.5 * np.sum(np.abs(
        target_probabilities - diagonal_model_probabilities
    ))

    print("\nA commuting model fitted only to Z-basis probabilities:")
    print(f"  fitted J           = {diagonal_coupling:.9f}")
    print("  target p(z)        =", np.array2string(
        target_probabilities, precision=9
    ))
    print("  commuting p(z)     =", np.array2string(
        diagonal_model_probabilities, precision=9
    ))
    print(f"  Z-basis TV distance = {total_variation:.3e}")
    print(f"  target average<X>   = {0.5 * target_statistics[1]:+.9f}")
    print(f"  commuting average<X>= {0.5 * sufficient_statistics(diagonal_model)[1]:+.9f}")
    print(
        "  quantum relative entropy = "
        f"{quantum_relative_entropy(target, diagonal_model):.9f}"
    )


if __name__ == "__main__":
    main()
