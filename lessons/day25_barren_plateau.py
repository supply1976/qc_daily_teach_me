"""第 25 課：global cost 如何造成 barren plateau。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


def product_state(parameters: np.ndarray) -> Statevector:
    """準備 tensor_i Ry(theta_i)|0>。"""
    circuit = QuantumCircuit(parameters.size)

    for qubit, angle in enumerate(parameters):
        circuit.ry(float(angle), qubit)

    return Statevector.from_instruction(circuit)


def costs(
    state: Statevector,
) -> tuple[float, float]:
    """回傳 global parity cost 與 q0 local cost。"""
    number_of_qubits = state.num_qubits
    global_observable = Pauli("Z" * number_of_qubits)

    # Qiskit label 從 q_(n-1) 寫到 q0，故最右邊是 q0。
    local_observable = Pauli(
        "I" * (number_of_qubits - 1) + "Z"
    )

    global_cost = state.expectation_value(global_observable)
    local_cost = state.expectation_value(local_observable)

    return (
        float(np.real_if_close(global_cost)),
        float(np.real_if_close(local_cost)),
    )


def parameter_shift_gradients(
    parameters: np.ndarray,
) -> tuple[float, float]:
    """用兩次 Qiskit statevector evaluations 求 theta_0 梯度。"""
    plus = parameters.copy()
    minus = parameters.copy()
    plus[0] += np.pi / 2
    minus[0] -= np.pi / 2

    costs_plus = costs(product_state(plus))
    costs_minus = costs(product_state(minus))

    return tuple(
        0.5 * (plus_value - minus_value)
        for plus_value, minus_value
        in zip(costs_plus, costs_minus)
    )


def analytic_gradients(
    parameter_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """向量化計算解析梯度，供大型 Monte Carlo 使用。"""
    theta_zero = parameter_batch[:, 0]

    global_gradient = (
        -np.sin(theta_zero)
        * np.prod(np.cos(parameter_batch[:, 1:]), axis=1)
    )
    local_gradient = -np.sin(theta_zero)

    return global_gradient, local_gradient


def main() -> None:
    rng = np.random.default_rng(2500)

    # 先確認 Qiskit parameter-shift 與解析式相同。
    maximum_error = 0.0

    for number_of_qubits in [2, 4, 6, 8]:
        for _ in range(10):
            parameters = rng.uniform(
                0.0,
                2 * np.pi,
                size=number_of_qubits,
            )
            qiskit_gradient = np.array(
                parameter_shift_gradients(parameters)
            )
            analytic_gradient = np.array([
                -np.sin(parameters[0])
                * np.prod(np.cos(parameters[1:])),
                -np.sin(parameters[0]),
            ])
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(
                    qiskit_gradient - analytic_gradient
                ))),
            )

    print(
        "Maximum Qiskit/analytic gradient error: "
        f"{maximum_error:.3e}"
    )

    # 大量隨機初始化用向量化解析式，避免建立數百萬個電路。
    trials = 200_000
    print(f"Random initializations per size: {trials}")
    print()
    print(
        " qubits   Var(global grad)   theory 2^-n   "
        "ratio    Var(local grad)"
    )
    print("---------------------------------------------------------------")

    for number_of_qubits in [2, 4, 6, 8, 10, 12, 14]:
        parameter_batch = rng.uniform(
            0.0,
            2 * np.pi,
            size=(trials, number_of_qubits),
        )
        global_gradient, local_gradient = (
            analytic_gradients(parameter_batch)
        )

        empirical_global_variance = float(
            np.var(global_gradient)
        )
        theoretical_global_variance = 2.0 ** (
            -number_of_qubits
        )
        empirical_local_variance = float(
            np.var(local_gradient)
        )

        print(
            f"{number_of_qubits:7d}   "
            f"{empirical_global_variance:16.8e}   "
            f"{theoretical_global_variance:11.8e}   "
            f"{empirical_global_variance / theoretical_global_variance:5.3f}   "
            f"{empirical_local_variance:15.8e}"
        )

    print("\nTypical global-gradient scale:")

    for number_of_qubits in [4, 8, 12, 16, 20]:
        standard_deviation = 2.0 ** (
            -number_of_qubits / 2
        )
        rough_shots = 2**number_of_qubits
        print(
            f"  n={number_of_qubits:2d}: "
            f"std={standard_deviation:.6e}, "
            f"rough shots for SNR~1 = {rough_shots:,}"
        )


if __name__ == "__main__":
    main()
