"""Day 03：Bloch sphere 與單 qubit 旋轉。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def main() -> None:
    theta = np.pi / 2
    phi = np.pi / 3

    # 忽略全域相位後：Ry(theta) 再 Rz(phi) 會準備
    # cos(theta/2)|0> + exp(i*phi)sin(theta/2)|1>。
    before_h = QuantumCircuit(1)
    before_h.ry(theta, 0)
    before_h.rz(phi, 0)
    state_before = Statevector.from_instruction(before_h)

    print("--- Before final H ---")
    print(before_h.draw())
    print("Statevector:", np.round(state_before.data, 6))
    print("Probabilities:", np.round(np.abs(state_before.data) ** 2, 6))

    # H 將相位差轉換成 computational-basis 的量測機率差。
    after_h = before_h.copy()
    after_h.h(0)
    state_after = Statevector.from_instruction(after_h)
    state_after.seed(42)

    print("\n--- After final H ---")
    print(after_h.draw())
    print("Statevector:", np.round(state_after.data, 6))
    print("Probabilities:", np.round(np.abs(state_after.data) ** 2, 6))
    print("1000 measurements:", state_after.sample_counts(shots=1000))


if __name__ == "__main__":
    main()

