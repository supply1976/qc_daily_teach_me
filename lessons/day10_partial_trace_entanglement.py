"""第 10 課：部分跡、約化密度矩陣與糾纏熵。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    entropy,
    mutual_information,
    partial_trace,
    purity,
)


def clean_number(value: complex, tolerance: float = 1e-12) -> float:
    """移除浮點計算產生的極小誤差，讓輸出較容易閱讀。"""
    real_value = float(np.real_if_close(value))
    return 0.0 if abs(real_value) < tolerance else real_value


def report_state(name: str, state: Statevector) -> None:
    """列出二 qubit 純態及其兩個單 qubit 約化態。"""
    global_state = DensityMatrix(state)

    # partial_trace 的 qargs 是「要消去」的 qubit：
    # 消去 q1 會留下 q0；消去 q0 會留下 q1。
    reduced_q0 = partial_trace(global_state, [1])
    reduced_q1 = partial_trace(global_state, [0])

    print(f"\n=== {name} ===")
    print("Statevector:")
    print(np.round(state.data, 6))

    print("Reduced density matrix of q0:")
    print(np.round(reduced_q0.data, 6))

    print("Reduced density matrix of q1:")
    print(np.round(reduced_q1.data, 6))

    print("Purity Tr(rho^2):")
    print(f"  global = {clean_number(purity(global_state)):.6f}")
    print(f"  q0     = {clean_number(purity(reduced_q0)):.6f}")
    print(f"  q1     = {clean_number(purity(reduced_q1)):.6f}")

    print("von Neumann entropy (bits):")
    print(f"  S(AB)  = {clean_number(entropy(global_state, base=2)):.6f}")
    print(f"  S(A)   = {clean_number(entropy(reduced_q0, base=2)):.6f}")
    print(f"  S(B)   = {clean_number(entropy(reduced_q1, base=2)):.6f}")

    information = mutual_information(global_state, base=2)
    print(f"Mutual information I(A:B) = {clean_number(information):.6f} bits")


def main() -> None:
    # 1. Bell state: (|00> + |11>) / sqrt(2)
    bell_circuit = QuantumCircuit(2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    bell_state = Statevector.from_instruction(bell_circuit)

    # 2. Product state: |+> tensor |+>
    # 兩個 qubit 各自有疊加，但彼此沒有糾纏。
    product_circuit = QuantumCircuit(2)
    product_circuit.h([0, 1])
    product_state = Statevector.from_instruction(product_circuit)

    print("Bell circuit:")
    print(bell_circuit.draw(output="text"))

    report_state("Bell state |Phi+>", bell_state)
    report_state("Product state |++>", product_state)


if __name__ == "__main__":
    main()
