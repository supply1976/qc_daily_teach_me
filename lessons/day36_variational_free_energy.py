"""Day 36: learn a Gibbs state by variational free-energy minimization."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    entropy,
    partial_trace,
    state_fidelity,
)


X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def hamiltonian(z_field, x_field):
    return z_field * Z + x_field * X


def purification_state(parameters):
    """Two-qubit pure state whose reduced system state can be mixed."""
    mixing_angle, basis_angle = parameters

    # q0 is the auxiliary qubit; q1 is the physical system.
    circuit = QuantumCircuit(2)
    circuit.ry(float(mixing_angle), 0)
    circuit.cx(0, 1)
    circuit.ry(float(basis_angle), 1)
    return Statevector.from_instruction(circuit)


def system_density_matrix(parameters):
    # Trace out auxiliary q0, retaining physical system q1.
    return partial_trace(purification_state(parameters), [0])


def energy(parameters, matrix):
    density = system_density_matrix(parameters)
    return float(np.real(np.trace(density.data @ matrix)))


def thermal_entropy(parameters):
    # Natural logarithms are required in F = E - S / beta.
    return float(entropy(system_density_matrix(parameters), base=np.e))


def free_energy(parameters, matrix, beta):
    return energy(parameters, matrix) - thermal_entropy(parameters) / beta


def parameter_shift_energy_gradient(parameters, matrix):
    gradient = np.zeros_like(parameters)

    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += np.pi / 2
        minus[index] -= np.pi / 2
        gradient[index] = 0.5 * (
            energy(plus, matrix) - energy(minus, matrix)
        )

    return gradient


def entropy_gradient(parameters, epsilon=1e-5):
    """Differentiate the nonlinear entropy term by central differences."""
    gradient = np.zeros_like(parameters)

    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (
            thermal_entropy(plus) - thermal_entropy(minus)
        ) / (2 * epsilon)

    return gradient


def free_energy_gradient(parameters, matrix, beta):
    return (
        parameter_shift_energy_gradient(parameters, matrix)
        - entropy_gradient(parameters) / beta
    )


def gibbs_state(matrix, beta):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    weights = np.exp(-beta * (eigenvalues - eigenvalues[0]))
    probabilities = weights / np.sum(weights)
    density = (
        eigenvectors
        @ np.diag(probabilities)
        @ eigenvectors.conj().T
    )
    return DensityMatrix(density)


def train(matrix, beta, steps=250, learning_rate=0.05):
    parameters = np.array([1.0, -0.5])
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    checkpoints = {0, 1, 10, 25, 50, 100, 200, steps}

    history = [(0, parameters.copy(), free_energy(parameters, matrix, beta))]

    for step in range(1, steps + 1):
        gradient = free_energy_gradient(parameters, matrix, beta)
        first_moment = beta_1 * first_moment + (1 - beta_1) * gradient
        second_moment = beta_2 * second_moment + (1 - beta_2) * gradient**2

        corrected_first = first_moment / (1 - beta_1**step)
        corrected_second = second_moment / (1 - beta_2**step)
        parameters -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )

        if step in checkpoints:
            history.append(
                (step, parameters.copy(), free_energy(parameters, matrix, beta))
            )

    return parameters, history


def main():
    z_field = 0.7
    x_field = 1.1
    beta = 1.0
    matrix = hamiltonian(z_field, x_field)
    omega = np.sqrt(z_field**2 + x_field**2)

    target = gibbs_state(matrix, beta)
    exact_free_energy = -np.log(2 * np.cosh(beta * omega)) / beta
    exact_basis_angle = np.arctan2(x_field, z_field)
    exact_ground_probability = 1 / (1 + np.exp(-2 * beta * omega))
    exact_mixing_angle = 2 * np.arcsin(
        np.sqrt(exact_ground_probability)
    )

    parameters, history = train(matrix, beta)

    print(f"Hamiltonian: H = {z_field} Z + {x_field} X")
    print(f"Inverse temperature beta: {beta:.3f}")
    print(f"Exact Gibbs free energy: {exact_free_energy:+.12f}\n")
    print(" step       free energy       gap = F-F*      beta*gap (= relative entropy)")
    print("-" * 75)

    for step, values, value in history:
        gap = value - exact_free_energy
        print(f"{step:5d}   {value:+.12f}   {gap:.3e}               {beta * gap:.3e}")

    learned = system_density_matrix(parameters)
    learned_energy = energy(parameters, matrix)
    learned_entropy = thermal_entropy(parameters)

    print("\nLearned parameters [mixing angle, basis angle]:")
    print(np.array2string(parameters, precision=9))
    print("Exact representative parameters:")
    print(np.array2string(
        np.array([exact_mixing_angle, exact_basis_angle]),
        precision=9,
    ))
    print("\nLearned density matrix:")
    print(np.array2string(learned.data, precision=9, suppress_small=True))
    print("\nExact Gibbs density matrix:")
    print(np.array2string(target.data, precision=9, suppress_small=True))
    print(f"\nLearned energy:          {learned_energy:+.12f}")
    print(f"Learned entropy:         {learned_entropy:.12f} nat")
    print(f"State fidelity:          {state_fidelity(learned, target):.12f}")
    print(
        "Maximum matrix error:    "
        f"{np.max(np.abs(learned.data - target.data)):.3e}"
    )


if __name__ == "__main__":
    main()
