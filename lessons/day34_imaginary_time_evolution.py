"""Day 34: Exact and variational quantum imaginary-time evolution."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def ansatz_state(theta: float) -> Statevector:
    """One-parameter state |psi(theta)> = Ry(theta)|0>."""
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    return Statevector.from_instruction(circuit)


def expectation(state: Statevector, operator: SparsePauliOp) -> float:
    value = state.expectation_value(operator)
    return float(np.real_if_close(value))


def energy_gradient(theta: float, hamiltonian: SparsePauliOp) -> float:
    """Exact parameter-shift derivative of <H>."""
    plus = expectation(
        ansatz_state(theta + np.pi / 2),
        hamiltonian,
    )
    minus = expectation(
        ansatz_state(theta - np.pi / 2),
        hamiltonian,
    )
    return 0.5 * (plus - minus)


def exact_imaginary_time_state(
    initial: Statevector,
    hamiltonian_matrix: np.ndarray,
    imaginary_time: float,
) -> Statevector:
    """Apply exp(-tau H), then renormalize the state."""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)

    # Subtract E0 to avoid exponential overflow. The removed scalar
    # disappears when the state is normalized.
    factors = np.exp(
        -imaginary_time * (eigenvalues - eigenvalues[0])
    )
    coefficients = eigenvectors.conj().T @ initial.data
    evolved = eigenvectors @ (factors * coefficients)
    evolved /= np.linalg.norm(evolved)
    return Statevector(evolved)


def exact_real_time_state(
    initial: Statevector,
    hamiltonian_matrix: np.ndarray,
    time: float,
) -> Statevector:
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
    phases = np.exp(-1j * time * eigenvalues)
    evolved = eigenvectors @ (
        phases * (eigenvectors.conj().T @ initial.data)
    )
    return Statevector(evolved)


def variational_qite(
    initial_theta: float,
    hamiltonian: SparsePauliOp,
    time_step: float,
    steps: int,
) -> np.ndarray:
    """Euler integration of theta_dot = -2 dE/dtheta."""
    trajectory = np.empty(steps + 1)
    trajectory[0] = initial_theta

    for step in range(steps):
        theta = trajectory[step]
        gradient = energy_gradient(theta, hamiltonian)

        # For Ry(theta)|0>, <partial_theta psi|partial_theta psi>=1/4.
        # McLachlan imaginary-time evolution gives theta_dot=-2*dE/dtheta.
        trajectory[step + 1] = (
            theta - 2.0 * time_step * gradient
        )

    return trajectory


def energy_variance(
    state: Statevector,
    hamiltonian_matrix: np.ndarray,
) -> float:
    mean = np.vdot(
        state.data,
        hamiltonian_matrix @ state.data,
    )
    second = np.vdot(
        state.data,
        hamiltonian_matrix @ hamiltonian_matrix @ state.data,
    )
    return float(np.real_if_close(second - mean * mean))


def fidelity(first: Statevector, second: Statevector) -> float:
    return float(abs(np.vdot(first.data, second.data)) ** 2)


def main() -> None:
    z_field = 0.7
    x_field = 1.1
    hamiltonian = SparsePauliOp.from_list([
        ("Z", z_field),
        ("X", x_field),
    ])
    matrix = hamiltonian.to_matrix()

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    ground_state = Statevector(eigenvectors[:, 0])
    initial_state = ansatz_state(0.0)

    time_step = 0.01
    maximum_time = 2.0
    steps = int(round(maximum_time / time_step))
    theta_trajectory = variational_qite(
        initial_theta=0.0,
        hamiltonian=hamiltonian,
        time_step=time_step,
        steps=steps,
    )

    print("Hamiltonian H = 0.7 Z + 1.1 X")
    print("Exact energies:", np.round(eigenvalues, 9))
    print(
        "Initial ground-state probability: "
        f"{fidelity(initial_state, ground_state):.9f}"
    )
    print(f"Variational QITE step: {time_step:.3f}")

    print("\nImaginary-time ground-state filtering:")
    print(
        " tau     exact E       variational E    "
        "ground prob.    state fidelity"
    )
    print("---------------------------------------------------------------")

    for imaginary_time in [0.0, 0.2, 0.5, 1.0, 2.0]:
        exact_state = exact_imaginary_time_state(
            initial_state,
            matrix,
            imaginary_time,
        )
        index = int(round(imaginary_time / time_step))
        variational_state = ansatz_state(theta_trajectory[index])

        print(
            f" {imaginary_time:3.1f}   "
            f"{expectation(exact_state, hamiltonian):+.9f}   "
            f"{expectation(variational_state, hamiltonian):+.9f}      "
            f"{fidelity(exact_state, ground_state):.9f}      "
            f"{fidelity(exact_state, variational_state):.9f}"
        )

    # Check dE/dtau = -2 Var(H) at tau=0.4 by a central difference.
    check_time = 0.4
    epsilon = 1e-5
    before = exact_imaginary_time_state(
        initial_state,
        matrix,
        check_time - epsilon,
    )
    after = exact_imaginary_time_state(
        initial_state,
        matrix,
        check_time + epsilon,
    )
    center = exact_imaginary_time_state(
        initial_state,
        matrix,
        check_time,
    )
    finite_difference = (
        expectation(after, hamiltonian)
        - expectation(before, hamiltonian)
    ) / (2.0 * epsilon)
    variance_identity = -2.0 * energy_variance(center, matrix)

    print("\nEnergy monotonicity identity at tau=0.4:")
    print(f"finite-difference dE/dtau: {finite_difference:+.9f}")
    print(f"-2 Var(H):                {variance_identity:+.9f}")

    real_time_state = exact_real_time_state(
        initial_state,
        matrix,
        time=2.0,
    )
    imaginary_time_state = exact_imaginary_time_state(
        initial_state,
        matrix,
        imaginary_time=2.0,
    )

    print("\nReal time versus imaginary time at t=tau=2:")
    print(
        "real-time energy:      "
        f"{expectation(real_time_state, hamiltonian):+.9f}"
    )
    print(
        "imaginary-time energy: "
        f"{expectation(imaginary_time_state, hamiltonian):+.9f}"
    )
    print(f"final variational theta: {theta_trajectory[-1]:+.9f}")


if __name__ == "__main__":
    main()
