"""第 24 課：用 SPSA 在 finite-shot noise 下訓練 VQE。"""

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
    circuit = QuantumCircuit(2)
    circuit.ry(parameters[0], 0)
    circuit.ry(parameters[1], 1)
    circuit.cx(0, 1)
    circuit.ry(parameters[2], 0)
    circuit.ry(parameters[3], 1)
    return Statevector.from_instruction(circuit)


def sign(bit: str) -> int:
    return 1 if bit == "0" else -1


def z_statistics(
    counts: dict[str, int],
) -> tuple[float, float, float]:
    """由 |q1 q0> counts 取得 Z0、Z1、Z0Z1 的樣本平均。"""
    shots = sum(counts.values())
    z0 = z1 = zz = 0.0

    for bits, count in counts.items():
        value_q0 = sign(bits[-1])
        value_q1 = sign(bits[-2])
        z0 += count * value_q0
        z1 += count * value_q1
        zz += count * value_q0 * value_q1

    return z0 / shots, z1 / shots, zz / shots


def noisy_energy(
    parameters: np.ndarray,
    coupling: float,
    field: float,
    shots: int,
    seed: int,
) -> float:
    """以 Z、X 兩個 measurement settings 估計 Ising energy。"""
    state = ansatz_state(parameters)

    state.seed(seed)
    z_counts = dict(state.sample_counts(shots))
    _, _, expectation_zz = z_statistics(z_counts)

    rotate_to_x_basis = QuantumCircuit(2)
    rotate_to_x_basis.h([0, 1])
    x_basis_state = state.evolve(rotate_to_x_basis)
    x_basis_state.seed(seed + 1)
    x_counts = dict(x_basis_state.sample_counts(shots))
    expectation_x0, expectation_x1, _ = z_statistics(x_counts)

    return float(
        -coupling * expectation_zz
        -field * (expectation_x0 + expectation_x1)
    )


def exact_energy(
    parameters: np.ndarray,
    hamiltonian: SparsePauliOp,
) -> float:
    value = ansatz_state(parameters).expectation_value(hamiltonian)
    return float(np.real_if_close(value))


def entanglement_entropy(state: Statevector) -> float:
    return float(entropy(partial_trace(state, [1]), base=2))


def main() -> None:
    coupling = 0.8
    field = 1.1
    shots = 2000
    iterations = 300

    hamiltonian = SparsePauliOp.from_list([
        ("ZZ", -coupling),
        ("IX", -field),
        ("XI", -field),
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(
        hamiltonian.to_matrix()
    )
    ground_energy = float(eigenvalues[0])
    ground_state = Statevector(eigenvectors[:, 0])

    parameters = np.array([0.2, -0.4, 0.1, 0.3])
    rng = np.random.default_rng(2400)

    # SPSA schedules: a_k 控制更新量，c_k 控制有限差分距離。
    a = 0.5
    capital_a = 20.0
    alpha = 0.7
    c = 0.15
    gamma = 0.2
    report_iterations = {1, 25, 50, 100, 150, 200, 300}

    print("Finite-shot SPSA VQE")
    print(f"Exact ground energy: {ground_energy:.12f}")
    print(
        "Initial exact energy: "
        f"{exact_energy(parameters, hamiltonian):.12f}"
    )
    print("\n iter   cost evals    diagnostic energy    exact error")
    print("----------------------------------------------------------")

    for step in range(iterations):
        iteration = step + 1
        learning_rate = a / (capital_a + iteration) ** alpha
        perturbation_size = c / iteration**gamma

        # 一個 random Rademacher direction 同時擾動所有參數。
        delta = rng.choice([-1.0, 1.0], size=parameters.size)

        energy_plus = noisy_energy(
            parameters + perturbation_size * delta,
            coupling,
            field,
            shots,
            seed=2400 + 10 * step,
        )
        energy_minus = noisy_energy(
            parameters - perturbation_size * delta,
            coupling,
            field,
            shots,
            seed=2402 + 10 * step,
        )

        # 因 delta_j 是 +/-1，所以 1/delta_j = delta_j。
        gradient = (
            (energy_plus - energy_minus)
            / (2.0 * perturbation_size)
            * delta
        )
        parameters -= learning_rate * gradient

        if iteration in report_iterations:
            diagnostic_energy = exact_energy(
                parameters,
                hamiltonian,
            )
            print(
                f"{iteration:5d}   "
                f"{2 * iteration:10d}   "
                f"{diagnostic_energy: .10f}   "
                f"{diagnostic_energy - ground_energy:.3e}"
            )

    final_state = ansatz_state(parameters)
    final_energy = exact_energy(parameters, hamiltonian)
    energy_evaluations = 2 * iterations
    settings_per_energy = 2
    measurement_circuits = energy_evaluations * settings_per_energy

    print("\nOptimized parameters:")
    print(np.round(parameters, 6))
    print(f"Final exact energy error: {final_energy - ground_energy:.3e}")
    print(
        "Ground-state fidelity: "
        f"{state_fidelity(final_state, ground_state):.9f}"
    )
    print(
        "Entanglement entropy:  "
        f"{entanglement_entropy(final_state):.9f} bit"
    )
    print("\nQuantum sampling cost:")
    print(f"  SPSA energy evaluations:       {energy_evaluations}")
    print(f"  Measurement-setting circuits:  {measurement_circuits}")
    print(f"  Total shots:                   {measurement_circuits * shots}")
    print(
        "  Parameter-shift evaluations "
        f"for the same steps: {2 * parameters.size * iterations}"
    )


if __name__ == "__main__":
    main()
