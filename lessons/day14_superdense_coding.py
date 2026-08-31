"""第 14 課：量子超密編碼、Bell basis 與兩個 classical bits。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


MESSAGES = [(0, 0), (0, 1), (1, 0), (1, 1)]


def encoded_bell_state(z_bit: int, x_bit: int) -> Statevector:
    """建立 Bell pair，並在 Alice 的 q0 上編碼 classical message zx。"""
    circuit = QuantumCircuit(2)

    # Alice(q0) 與 Bob(q1) 預先共享 |Phi+>。
    circuit.h(0)
    circuit.cx(0, 1)

    # 編碼 U_zx = Z^z X^x。電路中先 X、再 Z。
    if x_bit == 1:
        circuit.x(0)
    if z_bit == 1:
        circuit.z(0)

    return Statevector.from_instruction(circuit)


def superdense_circuit(z_bit: int, x_bit: int) -> QuantumCircuit:
    """建立完整 superdense-coding circuit。"""
    circuit = QuantumCircuit(2)

    # 1. 預共享 Bell pair。
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.barrier()

    # 2. Alice 編碼兩個 classical bits。
    if x_bit == 1:
        circuit.x(0)
    if z_bit == 1:
        circuit.z(0)
    circuit.barrier()

    # 此時 Alice 把 q0 傳給 Bob。

    # 3. Bob 將 Bell basis 解碼成 computational basis。
    circuit.cx(0, 1)
    circuit.h(0)

    return circuit


def gram_matrix(states: list[Statevector]) -> np.ndarray:
    """計算 G_ij = <psi_i|psi_j>。"""
    size = len(states)
    matrix = np.empty((size, size), dtype=complex)

    for i, state_i in enumerate(states):
        for j, state_j in enumerate(states):
            matrix[i, j] = np.vdot(state_i.data, state_j.data)

    return matrix


def main() -> None:
    shots = 1024

    encoded_states = [
        encoded_bell_state(z_bit, x_bit)
        for z_bit, x_bit in MESSAGES
    ]

    print("Message order: 00, 01, 10, 11")
    print("Absolute Gram matrix of encoded Bell states:")
    print(np.round(np.abs(gram_matrix(encoded_states)), 6))

    print("\nmessage zx   Pauli on q0   Qiskit output |q1q0>   counts")
    print("----------------------------------------------------------------")

    for index, (z_bit, x_bit) in enumerate(MESSAGES):
        circuit = superdense_circuit(z_bit, x_bit)
        decoded_state = Statevector.from_instruction(circuit)
        decoded_state.seed(1400 + index)
        counts = {
            str(outcome): int(count)
            for outcome, count in decoded_state.sample_counts(
                shots=shots
            ).items()
        }

        pauli = {
            (0, 0): "I",
            (0, 1): "X",
            (1, 0): "Z",
            (1, 1): "X then Z",
        }[(z_bit, x_bit)]

        # 解碼後 q0=z、q1=x；Qiskit 顯示順序為 |q1q0>=|xz>。
        qiskit_output = f"{x_bit}{z_bit}"
        message = f"{z_bit}{x_bit}"

        print(
            f"    {message}       {pauli:8s}             "
            f"{qiskit_output}            {counts}"
        )

    print("\nExample circuit for message zx=11:")
    print(superdense_circuit(1, 1).draw(output="text"))


if __name__ == "__main__":
    main()
