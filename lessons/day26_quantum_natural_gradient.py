"""第 26 課：Quantum Natural Gradient 與量子態空間幾何。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    SparsePauliOp,
    Statevector,
    state_fidelity,
)


def ansatz_state(parameters: np.ndarray) -> Statevector:
    """|psi(theta, phi)> = Rz(phi) Ry(theta) |0>."""
    theta, phi = parameters
    circuit = QuantumCircuit(1)
    circuit.ry(float(theta), 0)
    circuit.rz(float(phi), 0)
    return Statevector.from_instruction(circuit)


def energy(
    parameters: np.ndarray,
    hamiltonian: SparsePauliOp,
) -> float:
    value = ansatz_state(parameters).expectation_value(
        hamiltonian
    )
    return float(np.real_if_close(value))


def parameter_shift_gradient(
    parameters: np.ndarray,
    hamiltonian: SparsePauliOp,
) -> np.ndarray:
    """利用 parameter-shift 求 energy gradient。"""
    gradient = np.zeros_like(parameters)

    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += np.pi / 2
        minus[index] -= np.pi / 2
        gradient[index] = 0.5 * (
            energy(plus, hamiltonian)
            - energy(minus, hamiltonian)
        )

    return gradient


def state_derivatives(
    parameters: np.ndarray,
) -> list[np.ndarray]:
    """由 shifted statevectors 計算 |partial_i psi>."""
    derivatives = []

    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += np.pi / 2
        minus[index] -= np.pi / 2

        derivative = (
            ansatz_state(plus).data
            - ansatz_state(minus).data
        ) / (2 * np.sqrt(2))
        derivatives.append(derivative)

    return derivatives


def quantum_fisher_information(
    parameters: np.ndarray,
) -> np.ndarray:
    """計算 pure-state QFIM = 4 Re(QGT)。"""
    state = ansatz_state(parameters).data
    derivatives = state_derivatives(parameters)
    number_of_parameters = parameters.size
    qfim = np.zeros(
        (number_of_parameters, number_of_parameters)
    )

    for row in range(number_of_parameters):
        for column in range(number_of_parameters):
            geometric_tensor = (
                np.vdot(
                    derivatives[row],
                    derivatives[column],
                )
                - np.vdot(derivatives[row], state)
                * np.vdot(state, derivatives[column])
            )
            qfim[row, column] = 4 * float(
                np.real(geometric_tensor)
            )

    return qfim


def main() -> None:
    # Hamiltonian 的 ground-state Bloch vector 指向
    # (sin(theta*)cos(phi*), sin(theta*)sin(phi*), cos(theta*)).
    target_theta = 1.2
    target_phi = 1.4
    target_direction = np.array([
        np.sin(target_theta) * np.cos(target_phi),
        np.sin(target_theta) * np.sin(target_phi),
        np.cos(target_theta),
    ])

    hamiltonian = SparsePauliOp.from_list([
        ("X", -target_direction[0]),
        ("Y", -target_direction[1]),
        ("Z", -target_direction[2]),
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(
        hamiltonian.to_matrix()
    )
    ground_energy = float(eigenvalues[0])
    ground_state = Statevector(eigenvectors[:, 0])

    initial_parameters = np.array([0.3, -1.0])
    gradient_descent = initial_parameters.copy()
    natural_gradient = initial_parameters.copy()
    learning_rate = 0.2
    damping = 0.01
    report_steps = {0, 1, 2, 3, 5, 10, 20, 40}

    initial_qfim = quantum_fisher_information(
        initial_parameters
    )
    expected_qfim = np.diag([
        1.0,
        np.sin(initial_parameters[0]) ** 2,
    ])

    print("Initial quantum Fisher information matrix:")
    print(np.round(initial_qfim, 9))
    print(
        "Maximum error versus diag(1, sin(theta)^2): "
        f"{np.max(np.abs(initial_qfim - expected_qfim)):.3e}"
    )
    print(f"Exact ground energy: {ground_energy:.12f}")
    print()
    print(" step       ordinary GD       quantum natural GD")
    print("----------------------------------------------------")

    for step in range(41):
        if step in report_steps:
            print(
                f"{step:5d}   "
                f"{energy(gradient_descent, hamiltonian): .10f}   "
                f"{energy(natural_gradient, hamiltonian): .10f}"
            )

        if step == 40:
            break

        ordinary_gradient = parameter_shift_gradient(
            gradient_descent,
            hamiltonian,
        )
        gradient_descent -= (
            learning_rate * ordinary_gradient
        )

        natural_energy_gradient = parameter_shift_gradient(
            natural_gradient,
            hamiltonian,
        )
        metric = quantum_fisher_information(natural_gradient)
        natural_direction = np.linalg.solve(
            metric + damping * np.eye(2),
            natural_energy_gradient,
        )
        natural_gradient -= learning_rate * natural_direction

    ordinary_state = ansatz_state(gradient_descent)
    natural_state = ansatz_state(natural_gradient)

    print("\nFinal parameters [theta, phi]:")
    print("  ordinary GD:       ", np.round(gradient_descent, 6))
    print("  quantum natural GD:", np.round(natural_gradient, 6))
    print("Final ground-state fidelities:")
    print(
        "  ordinary GD:        "
        f"{state_fidelity(ordinary_state, ground_state):.12f}"
    )
    print(
        "  quantum natural GD: "
        f"{state_fidelity(natural_state, ground_state):.12f}"
    )


if __name__ == "__main__":
    main()
