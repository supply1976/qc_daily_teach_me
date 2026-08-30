"""第 13 課：量子隱形傳態、條件分支與 fidelity。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Statevector,
    partial_trace,
    purity,
    state_fidelity,
)


def prepare_unknown_state(theta: float, phi: float) -> Statevector:
    """準備 Rz(phi) Ry(theta)|0>，包含一般的複數相位。"""
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    circuit.rz(phi, 0)
    return Statevector.from_instruction(circuit)


def teleportation_prefix(theta: float, phi: float) -> QuantumCircuit:
    """建立到 Bell-basis measurement 之前的 teleportation circuit。"""
    circuit = QuantumCircuit(3)

    # q0：Alice 想傳送的未知量子態。
    circuit.ry(theta, 0)
    circuit.rz(phi, 0)
    circuit.barrier()

    # q1、q2：Alice 與 Bob 預先共享的 Bell pair。
    circuit.h(1)
    circuit.cx(1, 2)
    circuit.barrier()

    # 將 q0、q1 轉到 Bell measurement basis。
    circuit.cx(0, 1)
    circuit.h(0)

    return circuit


def conditional_bob_state(
    joint_state: Statevector,
    measured_q0: int,
    measured_q1: int,
) -> tuple[float, Statevector]:
    """由三 qubit statevector 擷取指定量測分支中的 Bob q2。"""
    # Qiskit statevector 使用 |q2 q1 q0> 排列。
    tensor = joint_state.data.reshape(2, 2, 2)
    branch = tensor[:, measured_q1, measured_q0]

    probability = float(np.vdot(branch, branch).real)
    normalized = branch / np.sqrt(probability)

    return probability, Statevector(normalized)


def correct_bob_state(
    state: Statevector,
    measured_q0: int,
    measured_q1: int,
) -> tuple[str, Statevector]:
    """根據 Alice 的兩個 classical bits 執行 X/Z corrections。"""
    correction = QuantumCircuit(1)
    gates = []

    if measured_q1 == 1:
        correction.x(0)
        gates.append("X")

    if measured_q0 == 1:
        correction.z(0)
        gates.append("Z")

    label = " then ".join(gates) if gates else "I"
    return label, state.evolve(correction)


def main() -> None:
    theta = 1.1
    phi = 0.7

    target = prepare_unknown_state(theta, phi)
    circuit = teleportation_prefix(theta, phi)
    joint_state = Statevector.from_instruction(circuit)

    print("Teleportation circuit before measurement:")
    print(circuit.draw(output="text"))
    print("Target |psi> amplitudes:")
    print(np.round(target.data, 6))

    print("\nAlice outcome   probability   correction   fidelity before/after")
    print("----------------------------------------------------------------")

    total_probability = 0.0

    for measured_q0 in [0, 1]:
        for measured_q1 in [0, 1]:
            probability, bob_before = conditional_bob_state(
                joint_state,
                measured_q0,
                measured_q1,
            )
            correction, bob_after = correct_bob_state(
                bob_before,
                measured_q0,
                measured_q1,
            )

            fidelity_before = state_fidelity(target, bob_before)
            fidelity_after = state_fidelity(target, bob_after)
            total_probability += probability

            outcome = f"m0m1={measured_q0}{measured_q1}"
            print(
                f"{outcome:13s}   {probability:.6f}      "
                f"{correction:8s}      "
                f"{fidelity_before:.6f} / {fidelity_after:.6f}"
            )

    print(f"\nSum of branch probabilities: {total_probability:.6f}")

    # Deferred-measurement principle：以 quantum-controlled gates 取代
    # measurement + classical feedforward，結果應完全相同。
    coherent_circuit = circuit.copy()
    coherent_circuit.barrier()
    coherent_circuit.cx(1, 2)
    coherent_circuit.cz(0, 2)

    coherent_state = Statevector.from_instruction(coherent_circuit)
    bob_density_matrix = partial_trace(coherent_state, [0, 1])
    bob_purity = float(np.real_if_close(purity(bob_density_matrix)))

    print("\nDeferred-measurement verification:")
    print("Bob reduced density matrix:")
    print(np.round(bob_density_matrix.data, 6))
    print(f"Bob purity:   {bob_purity:.6f}")
    print(f"Bob fidelity: {state_fidelity(target, bob_density_matrix):.6f}")


if __name__ == "__main__":
    main()
