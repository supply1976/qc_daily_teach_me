"""第 23 課：有限 shots 的 VQE energy estimation 與 Pauli grouping。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def ansatz_state(parameters: np.ndarray) -> Statevector:
    """重建第 22 課訓練完成的 four-parameter ansatz。"""
    circuit = QuantumCircuit(2)
    circuit.ry(parameters[0], 0)
    circuit.ry(parameters[1], 1)
    circuit.cx(0, 1)
    circuit.ry(parameters[2], 0)
    circuit.ry(parameters[3], 1)
    return Statevector.from_instruction(circuit)


def eigenvalue(bit: str) -> int:
    """Computational-basis bit 0/1 對應 Pauli-Z eigenvalue +1/-1。"""
    return 1 if bit == "0" else -1


def estimate_z_observables(
    counts: dict[str, int],
) -> tuple[float, float, float]:
    """由 |q1 q0> counts 同時估計 Z0、Z1、Z0Z1。"""
    shots = sum(counts.values())
    z0 = 0.0
    z1 = 0.0
    zz = 0.0

    for bits, count in counts.items():
        value_q0 = eigenvalue(bits[-1])
        value_q1 = eigenvalue(bits[-2])
        z0 += count * value_q0
        z1 += count * value_q1
        zz += count * value_q0 * value_q1

    return z0 / shots, z1 / shots, zz / shots


def sampled_energy(
    state: Statevector,
    coupling: float,
    field: float,
    shots: int,
    seed: int,
) -> tuple[float, tuple[float, float, float]]:
    """用兩個 measurement settings 估計三個 Pauli expectations。"""
    # Z-basis：直接量測 ZZ。
    state.seed(seed)
    z_counts = dict(state.sample_counts(shots=shots))
    _, _, expectation_zz = estimate_z_observables(z_counts)

    # X-basis：H 後量 Z，並由同一批 shots 同時取得 X0、X1。
    basis_change = QuantumCircuit(2)
    basis_change.h([0, 1])
    x_basis_state = state.evolve(basis_change)
    x_basis_state.seed(seed + 1)
    x_counts = dict(x_basis_state.sample_counts(shots=shots))
    expectation_x0, expectation_x1, _ = estimate_z_observables(
        x_counts
    )

    estimate = (
        -coupling * expectation_zz
        -field * (expectation_x0 + expectation_x1)
    )
    observables = (
        expectation_zz,
        expectation_x0,
        expectation_x1,
    )
    return estimate, observables


def main() -> None:
    coupling = 0.8
    field = 1.1

    hamiltonian = SparsePauliOp.from_list([
        ("ZZ", -coupling),
        ("IX", -field),
        ("XI", -field),
    ])

    # 第 22 課 VQE 得到的參數。
    parameters = np.array([
        0.375511,
        0.371524,
        1.428606,
        1.174169,
    ])
    state = ansatz_state(parameters)
    exact_energy = float(
        np.real_if_close(state.expectation_value(hamiltonian))
    )

    print("Finite-shot VQE energy estimation")
    print(f"Exact statevector energy: {exact_energy:.12f}")

    example_energy, example_observables = sampled_energy(
        state,
        coupling,
        field,
        shots=1000,
        seed=2300,
    )
    zz, x0, x1 = example_observables
    print("\nOne 1000-shot estimate per measurement setting:")
    print(f"  <ZZ> = {zz:+.6f}")
    print(f"  <X0> = {x0:+.6f}")
    print(f"  <X1> = {x1:+.6f}")
    print(f"  E_hat = {example_energy:+.6f}")

    trials = 200
    print(f"\nRepeated estimates ({trials} trials):")
    print(" shots      mean energy      bias       RMSE     RMSE*sqrt(N)")
    print("----------------------------------------------------------------")

    for shots in [100, 400, 1600, 6400]:
        estimates = np.array([
            sampled_energy(
                state,
                coupling,
                field,
                shots=shots,
                seed=23_000 + 2 * trial,
            )[0]
            for trial in range(trials)
        ])

        errors = estimates - exact_energy
        bias = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors**2)))

        print(
            f"{shots:6d}   "
            f"{np.mean(estimates): .9f}   "
            f"{bias:+.3e}   "
            f"{rmse:.3e}      "
            f"{rmse * np.sqrt(shots):.3f}"
        )

    print("\nEach row uses two settings, so total shots = 2 * shots.")


if __name__ == "__main__":
    main()
