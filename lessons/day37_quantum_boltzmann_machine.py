"""Day 37: train a two-qubit diagonal quantum Boltzmann machine."""

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp


# Qiskit computational-basis order: |q1 q0> = 00, 01, 10, 11.
Z0_VALUES = np.array([1.0, -1.0, 1.0, -1.0])
Z1_VALUES = np.array([1.0, 1.0, -1.0, -1.0])
ZZ_VALUES = Z0_VALUES * Z1_VALUES
FEATURES = np.stack([Z0_VALUES, Z1_VALUES, ZZ_VALUES], axis=1)


def qbm_hamiltonian(parameters):
    """H = -b0 Z0 - b1 Z1 - J Z0 Z1."""
    bias_0, bias_1, coupling = parameters
    return SparsePauliOp.from_list([
        ("IZ", -float(bias_0)),
        ("ZI", -float(bias_1)),
        ("ZZ", -float(coupling)),
    ])


def gibbs_distribution(parameters, beta=1.0):
    matrix = qbm_hamiltonian(parameters).to_matrix()
    energies = np.real(np.diag(matrix))

    # Subtracting the minimum energy improves numerical stability.
    weights = np.exp(-beta * (energies - np.min(energies)))
    return weights / np.sum(weights)


def gibbs_density_matrix(parameters, beta=1.0):
    probabilities = gibbs_distribution(parameters, beta)
    return DensityMatrix(np.diag(probabilities.astype(complex)))


def moments(probabilities):
    """Return <Z0>, <Z1>, and <Z0 Z1>."""
    return probabilities @ FEATURES


def cross_entropy(target, model):
    return float(-np.sum(target * np.log(np.clip(model, 1e-15, 1.0))))


def kl_divergence(target, model):
    mask = target > 0
    return float(np.sum(
        target[mask] * np.log(target[mask] / model[mask])
    ))


def total_variation_distance(target, model):
    return float(0.5 * np.sum(np.abs(target - model)))


def likelihood_gradient(parameters, target, beta=1.0):
    """Exact gradient: beta * (model moments - data moments)."""
    model = gibbs_distribution(parameters, beta)
    return beta * (moments(model) - moments(target))


def train_qbm(target, beta=1.0, steps=200, learning_rate=0.08):
    parameters = np.array([0.3, -0.2, 0.0])
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    epsilon = 1e-8
    checkpoints = {0, 1, 10, 25, 50, 100, steps}
    history = []

    for step in range(steps + 1):
        model = gibbs_distribution(parameters, beta)

        if step in checkpoints:
            history.append((
                step,
                parameters.copy(),
                cross_entropy(target, model),
                kl_divergence(target, model),
            ))

        if step == steps:
            break

        gradient = likelihood_gradient(parameters, target, beta)
        iteration = step + 1
        first_moment = (
            adam_beta_1 * first_moment
            + (1 - adam_beta_1) * gradient
        )
        second_moment = (
            adam_beta_2 * second_moment
            + (1 - adam_beta_2) * gradient**2
        )
        corrected_first = first_moment / (1 - adam_beta_1**iteration)
        corrected_second = second_moment / (1 - adam_beta_2**iteration)
        parameters -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )

    return parameters, history


def sample_density_matrix(density, shots, seed):
    density.seed(seed)
    raw_counts = density.sample_counts(shots)

    # Include missing outcomes and return ordinary Python integers.
    return {
        format(index, "02b"): int(raw_counts.get(format(index, "02b"), 0))
        for index in range(4)
    }


def main():
    beta = 1.0
    target = np.array([0.45, 0.05, 0.05, 0.45])
    labels = ["00", "01", "10", "11"]

    parameters, history = train_qbm(target, beta)
    learned = gibbs_distribution(parameters, beta)
    density = gibbs_density_matrix(parameters, beta)

    print("Target distribution [00, 01, 10, 11]:")
    print(target)
    print("Target moments [<Z0>, <Z1>, <Z0 Z1>]:")
    print(moments(target))
    print()
    print(" step    cross entropy       KL(target || model)       b0          b1          J")
    print("-" * 87)

    for step, values, loss, divergence in history:
        print(
            f"{step:5d}   {loss:.12f}      {divergence:.3e}       "
            f"{values[0]:+.6f}   {values[1]:+.6f}   {values[2]:+.6f}"
        )

    exact_coupling = 0.5 * np.log(target[0] / target[1]) / beta
    generated_counts = sample_density_matrix(density, 10000, seed=3700)

    print("\nLearned distribution:")
    for label, probability in zip(labels, learned):
        print(f"  p({label}) = {probability:.9f}")

    print("\nLearned moments [<Z0>, <Z1>, <Z0 Z1>]:")
    print(np.array2string(moments(learned), precision=9))
    print(f"\nLearned coupling J:       {parameters[2]:.9f}")
    print(f"Exact symmetric coupling: {exact_coupling:.9f}")
    print(
        "Total variation distance: "
        f"{total_variation_distance(target, learned):.3e}"
    )
    print("Gibbs density-matrix purity: " f"{np.trace(density.data @ density.data).real:.9f}")
    print(f"Generated counts (10000 shots): {generated_counts}")


if __name__ == "__main__":
    main()
