"""Day 07：密度矩陣、量子 coherence 與古典混合。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector


def main() -> None:
    # --------------------------------------------------
    # 1. 準備 Bell 純態 (|00> + |11>) / sqrt(2)
    # --------------------------------------------------
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    bell_state = Statevector.from_instruction(circuit)
    rho_bell = DensityMatrix(bell_state)

    # --------------------------------------------------
    # 2. 建立古典混合：50% |00><00| + 50% |11><11|
    # --------------------------------------------------
    ket_00 = np.array([1, 0, 0, 0], dtype=complex)
    ket_11 = np.array([0, 0, 0, 1], dtype=complex)
    rho_mixed_data = (
        0.5 * np.outer(ket_00, ket_00.conj())
        + 0.5 * np.outer(ket_11, ket_11.conj())
    )
    rho_mixed = DensityMatrix(rho_mixed_data)

    # --------------------------------------------------
    # 3. 比較矩陣、純度與不同 basis 的相關性
    # --------------------------------------------------
    zz = Pauli("ZZ")
    xx = Pauli("XX")

    print("Bell-state density matrix:")
    print(np.round(rho_bell.data, 3))

    print("\nClassical-mixture density matrix:")
    print(np.round(rho_mixed.data, 3))

    print("\nComputational-basis probabilities:")
    print("Bell:   ", rho_bell.probabilities_dict())
    print("Mixture:", rho_mixed.probabilities_dict())

    print("\nPurity Tr(rho^2):")
    print("Bell:   ", np.real_if_close(rho_bell.purity()))
    print("Mixture:", np.real_if_close(rho_mixed.purity()))

    print("\nCorrelations:")
    print(
        "Bell    <ZZ>, <XX> =",
        np.real_if_close(rho_bell.expectation_value(zz)),
        np.real_if_close(rho_bell.expectation_value(xx)),
    )
    print(
        "Mixture <ZZ>, <XX> =",
        np.real_if_close(rho_mixed.expectation_value(zz)),
        np.real_if_close(rho_mixed.expectation_value(xx)),
    )


if __name__ == "__main__":
    main()
