"""Day 40: shot-based multi-basis training of a quantum Boltzmann machine."""

import numpy as np
from qiskit import QuantumCircuit
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


def exact_statistics(density):
    return np.array([
        expectation(density, "ZZ"),
        expectation(density, "IX") + expectation(density, "XI"),
    ])


def measurement_distribution(density, basis):
    """Return probabilities after the requested basis rotation."""
    if basis == "Z":
        rotated = density
    elif basis == "X":
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])
        rotated = density.evolve(circuit)
    else:
        raise ValueError("basis must be 'Z' or 'X'")

    probabilities = np.real(np.diag(rotated.data))
    return np.clip(probabilities, 0.0, 1.0) / np.sum(probabilities)


def sample_observable(density, basis, shots, seed):
    """Sample ZZ in Z basis or X0+X1 in X basis."""
    rng = np.random.default_rng(seed)
    outcomes = rng.choice(
        4,
        size=shots,
        p=measurement_distribution(density, basis),
    )
    q0 = outcomes & 1
    q1 = (outcomes >> 1) & 1

    if basis == "Z":
        samples = (1 - 2 * q0) * (1 - 2 * q1)
    else:
        samples = (1 - 2 * q0) + (1 - 2 * q1)

    estimate = float(np.mean(samples))
    standard_error = float(np.std(samples, ddof=1) / np.sqrt(shots))
    return estimate, standard_error


def sampled_statistics(density, shots, seed):
    zz, zz_error = sample_observable(
        density, "Z", shots, seed
    )
    x_sum, x_error = sample_observable(
        density, "X", shots, seed + 1
    )
    return np.array([zz, x_sum]), np.array([zz_error, x_error])


def matrix_log_positive(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-15, None)
    return (
        eigenvectors
        @ np.diag(np.log(eigenvalues))
        @ eigenvectors.conj().T
    )


def relative_entropy(target, model):
    value = np.trace(
        target.data
        @ (
            matrix_log_positive(target.data)
            - matrix_log_positive(model.data)
        )
    )
    return float(np.real(value))


def train_with_shots(
    target_statistics,
    initial,
    shots,
    steps=300,
    learning_rate=0.035,
    seed=4000,
):
    parameters = np.array(initial, dtype=float)
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    tail = []

    for step in range(steps):
        model = gibbs_state(parameters)
        model_statistics, _ = sampled_statistics(
            model,
            shots,
            seed + 10 * step,
        )
        gradient = BETA * (model_statistics - target_statistics)

        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1 - 0.9 ** (step + 1))
        corrected_second = second_moment / (1 - 0.999 ** (step + 1))
        parameters -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )

        if step >= steps - 50:
            tail.append(parameters.copy())

    # Polyak averaging reduces the final stochastic fluctuation.
    return np.mean(tail, axis=0)


def main():
    target_parameters = np.array([0.9, 0.65])
    target = gibbs_state(target_parameters)
    exact_target = exact_statistics(target)

    target_statistics, target_errors = sampled_statistics(
        target,
        shots=100_000,
        seed=3900,
    )

    print("Target state:")
    print(f"  exact <ZZ>          = {exact_target[0]:+.9f}")
    print(f"  sampled <ZZ>        = {target_statistics[0]:+.9f} +/- {target_errors[0]:.3e}")
    print(f"  exact average <X>   = {0.5 * exact_target[1]:+.9f}")
    print(
        "  sampled average <X> = "
        f"{0.5 * target_statistics[1]:+.9f} +/- {0.5 * target_errors[1]:.3e}"
    )

    print("\nShot-based stochastic QBM training:")
    print(" shots        J          Gamma      rel. entropy      fidelity")
    print("-" * 68)

    for shots in [256, 1024, 4096]:
        learned = train_with_shots(
            target_statistics,
            initial=np.array([-0.35, 0.20]),
            shots=shots,
            seed=4000 + shots,
        )
        learned_state = gibbs_state(learned)
        print(
            f"{shots:6d}   {learned[0]:+.6f}   {learned[1]:+.6f}"
            f"   {relative_entropy(target, learned_state):.6e}"
            f"   {state_fidelity(target, learned_state):.9f}"
        )

    print("\nRepeated estimates at the exact target state (shots=1024):")
    estimates = []
    reported_errors = []
    for repeat in range(200):
        estimate, errors = sampled_statistics(
            target,
            shots=1024,
            seed=9000 + 2 * repeat,
        )
        estimates.append(estimate)
        reported_errors.append(errors)

    estimates = np.asarray(estimates)
    reported_errors = np.asarray(reported_errors)
    empirical_std = np.std(estimates, axis=0, ddof=1)
    mean_reported_error = np.mean(reported_errors, axis=0)
    print(
        f"  empirical std of <ZZ>: {empirical_std[0]:.6f}, "
        f"mean predicted SE: {mean_reported_error[0]:.6f}"
    )
    print(
        f"  empirical std of average<X>: {0.5 * empirical_std[1]:.6f}, "
        f"mean predicted SE: {0.5 * mean_reported_error[1]:.6f}"
    )


if __name__ == "__main__":
    main()
