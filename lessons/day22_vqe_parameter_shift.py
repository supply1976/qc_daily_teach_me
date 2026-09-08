"""第 22 課：用 VQE 與 parameter-shift rule 尋找 Ising 基態。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    SparsePauliOp,
    Statevector,
    entropy,
    partial_trace,
    state_fidelity,
)


def ansatz_state(parameters: np.ndarray) -> Statevector:
    """建立含四個可訓練角度的 two-qubit trial state。"""
    if parameters.shape != (4,):
        raise ValueError("parameters must have shape (4,)")

    circuit = QuantumCircuit(2)
    circuit.ry(parameters[0], 0)
    circuit.ry(parameters[1], 1)
    circuit.cx(0, 1)
    circuit.ry(parameters[2], 0)
    circuit.ry(parameters[3], 1)

    return Statevector.from_instruction(circuit)


def energy(
    parameters: np.ndarray,
    hamiltonian: SparsePauliOp,
) -> float:
    """VQE loss: E(theta) = <psi(theta)|H|psi(theta)>."""
    state = ansatz_state(parameters)
    value = state.expectation_value(hamiltonian)
    return float(np.real_if_close(value))


def parameter_shift_gradient(
    parameters: np.ndarray,
    hamiltonian: SparsePauliOp,
) -> np.ndarray:
    """每個 RY 參數以兩次 energy evaluation 計算精確梯度。"""
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


def entanglement_entropy(state: Statevector) -> float:
    """Pure two-qubit state 的單 qubit entropy 即糾纏熵。"""
    reduced_q0 = partial_trace(state, [1])
    return float(entropy(reduced_q0, base=2))


def exact_ground_state(
    hamiltonian: SparsePauliOp,
) -> tuple[float, Statevector]:
    """小系統可用 NumPy diagonalization 取得驗證答案。"""
    eigenvalues, eigenvectors = np.linalg.eigh(
        hamiltonian.to_matrix()
    )
    return float(eigenvalues[0]), Statevector(eigenvectors[:, 0])


def main() -> None:
    coupling = 0.8
    field = 1.1

    # H = -J Z0 Z1 - h (X0 + X1)
    # Qiskit Pauli labels 採 |q1 q0> 排列。
    hamiltonian = SparsePauliOp.from_list([
        ("ZZ", -coupling),
        ("IX", -field),
        ("XI", -field),
    ])

    exact_energy, ground_state = exact_ground_state(
        hamiltonian
    )

    # 固定初始值使範例可以完全重現。
    parameters = np.array([0.2, -0.4, 0.1, 0.3])

    # 手寫 Adam；在硬體上，energy() 會由 finite-shot measurements 估計。
    learning_rate = 0.08
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)

    report_iterations = {1, 10, 25, 50, 100, 150, 200}

    print("Two-qubit transverse-field Ising VQE")
    print(f"Exact ground energy: {exact_energy:.12f}")
    print(f"Initial VQE energy:  {energy(parameters, hamiltonian):.12f}")
    print("\n iter       energy         error        |gradient|")
    print("---------------------------------------------------")

    for iteration in range(1, 201):
        gradient = parameter_shift_gradient(
            parameters,
            hamiltonian,
        )

        first_moment = (
            beta_1 * first_moment
            + (1.0 - beta_1) * gradient
        )
        second_moment = (
            beta_2 * second_moment
            + (1.0 - beta_2) * gradient**2
        )

        corrected_first = first_moment / (
            1.0 - beta_1**iteration
        )
        corrected_second = second_moment / (
            1.0 - beta_2**iteration
        )

        parameters -= (
            learning_rate
            * corrected_first
            / (np.sqrt(corrected_second) + epsilon)
        )

        if iteration in report_iterations:
            current_energy = energy(parameters, hamiltonian)
            print(
                f"{iteration:5d}   "
                f"{current_energy: .10f}   "
                f"{current_energy - exact_energy:.3e}   "
                f"{np.linalg.norm(gradient):.3e}"
            )

    optimized_state = ansatz_state(parameters)
    optimized_energy = energy(parameters, hamiltonian)

    print("\nOptimized parameters:")
    print(np.round(parameters, 6))
    print(f"Final energy error: {optimized_energy - exact_energy:.3e}")
    print(
        "Ground-state fidelity: "
        f"{state_fidelity(optimized_state, ground_state):.12f}"
    )
    print(
        "VQE entanglement entropy:   "
        f"{entanglement_entropy(optimized_state):.9f} bit"
    )
    print(
        "Exact entanglement entropy: "
        f"{entanglement_entropy(ground_state):.9f} bit"
    )
    print("VQE basis probabilities:")
    print({
        str(outcome): round(float(probability), 6)
        for outcome, probability
        in optimized_state.probabilities_dict().items()
        if probability > 1e-12
    })


if __name__ == "__main__":
    main()
