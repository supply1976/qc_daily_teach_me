"""Day 09：由 X、Y、Z 量測重建單 qubit 密度矩陣。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector


def estimate_pauli(
    state: DensityMatrix,
    axis: str,
    shots: int,
    seed: int,
) -> tuple[float, dict]:
    """用 basis rotation 加 Z-basis sampling 估計 Pauli expectation。"""
    axis = axis.upper()
    rotation = QuantumCircuit(1)

    if axis == "X":
        rotation.h(0)
    elif axis == "Y":
        rotation.sdg(0)
        rotation.h(0)
    elif axis != "Z":
        raise ValueError("axis must be X, Y, or Z")

    rotated_state = state.evolve(rotation)
    rotated_state.seed(seed)
    counts = rotated_state.sample_counts(shots=shots)

    n0 = int(counts.get("0", 0))
    n1 = int(counts.get("1", 0))
    expectation = (n0 - n1) / shots
    return expectation, dict(counts)


def density_from_bloch(x: float, y: float, z: float) -> DensityMatrix:
    """由 Bloch vector r=(x,y,z) 做 linear-inversion tomography。"""
    identity = np.eye(2, dtype=complex)
    matrix = 0.5 * (
        identity
        + x * Pauli("X").to_matrix()
        + y * Pauli("Y").to_matrix()
        + z * Pauli("Z").to_matrix()
    )
    return DensityMatrix(matrix)


def main() -> None:
    shots = 4000
    theta = np.pi / 3
    phi = np.pi / 4
    mixing = 0.30

    # 準備一個具有複數相位的純態。
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    circuit.rz(phi, 0)
    pure_state = DensityMatrix(Statevector.from_instruction(circuit))

    # 加入 30% 最大混合態，使待測狀態成為一般 mixed state。
    true_state = DensityMatrix(
        (1.0 - mixing) * pure_state.data
        + mixing * np.eye(2, dtype=complex) / 2.0
    )

    exact = {
        axis: float(np.real_if_close(true_state.expectation_value(Pauli(axis))))
        for axis in "XYZ"
    }

    estimated = {}
    print("Axis   exact      estimate    counts")
    print("-----------------------------------------------")
    for seed, axis in enumerate("XYZ", start=100):
        value, counts = estimate_pauli(true_state, axis, shots, seed)
        estimated[axis] = value
        print(f" {axis}    {exact[axis]: .6f}   {value: .6f}   {counts}")

    reconstructed = density_from_bloch(
        estimated["X"],
        estimated["Y"],
        estimated["Z"],
    )

    print("\nTrue density matrix:")
    print(np.round(true_state.data, 6))

    print("\nReconstructed density matrix:")
    print(np.round(reconstructed.data, 6))

    print("\nFrobenius reconstruction error:")
    print(np.linalg.norm(reconstructed.data - true_state.data))

    print("\nReconstructed eigenvalues:")
    print(np.round(np.linalg.eigvalsh(reconstructed.data), 6))


if __name__ == "__main__":
    main()
