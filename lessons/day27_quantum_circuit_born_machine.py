"""第 27 課：Quantum Circuit Born Machine 學習相關分布。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Statevector,
    entropy,
    partial_trace,
)


def qcbm_state(
    parameters: np.ndarray,
    entangle: bool = True,
) -> Statevector:
    """四參數 two-qubit Born machine。"""
    circuit = QuantumCircuit(2)
    circuit.ry(float(parameters[0]), 0)
    circuit.ry(float(parameters[1]), 1)

    if entangle:
        circuit.cx(0, 1)

    circuit.ry(float(parameters[2]), 0)
    circuit.ry(float(parameters[3]), 1)
    return Statevector.from_instruction(circuit)


def model_probabilities(
    parameters: np.ndarray,
    entangle: bool = True,
) -> np.ndarray:
    return qcbm_state(parameters, entangle).probabilities()


def cross_entropy(
    target: np.ndarray,
    model: np.ndarray,
) -> float:
    safe_model = np.clip(model, 1e-12, 1.0)
    return float(-np.sum(target * np.log(safe_model)))


def kl_divergence(
    target: np.ndarray,
    model: np.ndarray,
) -> float:
    safe_model = np.clip(model, 1e-12, 1.0)
    return float(np.sum(target * np.log(target / safe_model)))


def parameter_shift_loss_gradient(
    parameters: np.ndarray,
    target: np.ndarray,
    entangle: bool,
) -> np.ndarray:
    """由 shifted output probabilities 計算 cross-entropy gradient。"""
    model = np.clip(
        model_probabilities(parameters, entangle),
        1e-12,
        1.0,
    )
    gradient = np.zeros_like(parameters)

    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += np.pi / 2
        minus[index] -= np.pi / 2

        probability_derivative = 0.5 * (
            model_probabilities(plus, entangle)
            - model_probabilities(minus, entangle)
        )
        gradient[index] = -np.sum(
            target / model * probability_derivative
        )

    return gradient


def train_born_machine(
    target: np.ndarray,
    entangle: bool,
    report: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """以 Adam 最小化 target-to-model cross entropy。"""
    parameters = np.array([0.2, -0.4, 0.1, 0.3])
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    learning_rate = 0.05
    beta_1 = 0.9
    beta_2 = 0.999
    report_steps = {1, 10, 50, 100, 200}

    for step in range(1, 201):
        gradient = parameter_shift_loss_gradient(
            parameters,
            target,
            entangle,
        )
        first_moment = (
            beta_1 * first_moment
            + (1 - beta_1) * gradient
        )
        second_moment = (
            beta_2 * second_moment
            + (1 - beta_2) * gradient**2
        )

        corrected_first = first_moment / (1 - beta_1**step)
        corrected_second = second_moment / (1 - beta_2**step)
        parameters -= (
            learning_rate
            * corrected_first
            / (np.sqrt(corrected_second) + 1e-8)
        )

        if report and step in report_steps:
            model = model_probabilities(parameters, entangle)
            print(
                f"{step:5d}   "
                f"{cross_entropy(target, model):.10f}   "
                f"{kl_divergence(target, model):.3e}"
            )

    return parameters, model_probabilities(parameters, entangle)


def classical_mutual_information(
    probabilities: np.ndarray,
) -> float:
    """計算量測後兩個 classical bits 的 mutual information。"""
    joint = probabilities.reshape(2, 2)
    marginal_q0 = np.sum(joint, axis=0)
    marginal_q1 = np.sum(joint, axis=1)
    information = 0.0

    for q1 in range(2):
        for q0 in range(2):
            probability = joint[q1, q0]

            if probability > 0:
                information += probability * np.log2(
                    probability
                    / (marginal_q1[q1] * marginal_q0[q0])
                )

    return float(information)


def total_variation_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(0.5 * np.sum(np.abs(first - second)))


def main() -> None:
    # Array indices follow |q1 q0>: 00, 01, 10, 11.
    target = np.array([0.45, 0.05, 0.05, 0.45])

    print("Target distribution [00, 01, 10, 11]:")
    print(target)
    print(
        "Target classical mutual information: "
        f"{classical_mutual_information(target):.9f} bit"
    )
    print("\nEntangled QCBM training:")
    print(" step   cross entropy      KL(target || model)")
    print("------------------------------------------------")

    parameters, learned = train_born_machine(
        target,
        entangle=True,
        report=True,
    )
    product_parameters, product_model = train_born_machine(
        target,
        entangle=False,
    )

    learned_state = qcbm_state(parameters, entangle=True)
    entanglement = float(
        entropy(partial_trace(learned_state, [1]), base=2)
    )

    learned_state.seed(2700)
    counts = {
        str(outcome): int(count)
        for outcome, count
        in learned_state.sample_counts(shots=10_000).items()
    }

    print("\nLearned QCBM distribution:")
    print(np.round(learned, 8))
    print(
        "KL divergence:            "
        f"{kl_divergence(target, learned):.3e} nat"
    )
    print(
        "Total variation distance: "
        f"{total_variation_distance(target, learned):.3e}"
    )
    print(
        "Classical mutual information: "
        f"{classical_mutual_information(learned):.9f} bit"
    )
    print(f"Quantum entanglement entropy: {entanglement:.9f} bit")
    print(f"Generated counts (10000 shots): {counts}")

    print("\nProduct-state baseline (CX removed):")
    print(np.round(product_model, 8))
    print(
        "KL divergence:            "
        f"{kl_divergence(target, product_model):.9f} nat"
    )
    print(
        "Total variation distance: "
        f"{total_variation_distance(target, product_model):.9f}"
    )
    print("Optimized product parameters:")
    print(np.round(product_parameters, 6))


if __name__ == "__main__":
    main()
