"""第 12 課：CHSH inequality、Bell nonlocality 與有限 shots。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def bell_state() -> tuple[QuantumCircuit, Statevector]:
    """建立 |Phi+> = (|00> + |11>) / sqrt(2)。"""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit, Statevector.from_instruction(circuit)


def correlation_operator(theta_a: float, theta_b: float) -> SparsePauliOp:
    """建立 A(theta_a) tensor B(theta_b) 的 Pauli 展開。"""
    # O(theta) = cos(theta) Z + sin(theta) X。
    # Qiskit Pauli label 以 |q1 q0> 排列，因此 B 在左、A 在右。
    return SparsePauliOp.from_list(
        [
            ("ZZ", np.cos(theta_b) * np.cos(theta_a)),
            ("ZX", np.cos(theta_b) * np.sin(theta_a)),
            ("XZ", np.sin(theta_b) * np.cos(theta_a)),
            ("XX", np.sin(theta_b) * np.sin(theta_a)),
        ]
    ).simplify()


def exact_correlation(
    state: Statevector,
    theta_a: float,
    theta_b: float,
) -> float:
    """用 statevector expectation value 計算精確 correlation。"""
    operator = correlation_operator(theta_a, theta_b)
    value = state.expectation_value(operator)
    return float(np.real_if_close(value))


def sampled_correlation(
    state: Statevector,
    theta_a: float,
    theta_b: float,
    shots: int,
    seed: int,
) -> tuple[float, dict[str, int]]:
    """旋轉量測基底，再由有限 shots 估計 correlation。"""
    rotation = QuantumCircuit(2)

    # Ry(-theta) 後量 Z，等價於量 cos(theta)Z + sin(theta)X。
    rotation.ry(-theta_a, 0)
    rotation.ry(-theta_b, 1)

    rotated_state = state.evolve(rotation)
    rotated_state.seed(seed)
    counts = dict(rotated_state.sample_counts(shots=shots))

    # 00、11 的 eigenvalue product 是 +1；01、10 是 -1。
    same = counts.get("00", 0) + counts.get("11", 0)
    different = counts.get("01", 0) + counts.get("10", 0)
    correlation = (same - different) / shots

    return correlation, counts


def chsh_value(state: Statevector) -> float:
    """精確計算 S = E00 + E01 + E10 - E11。"""
    settings = [
        (0.0, np.pi / 4, +1.0),
        (0.0, -np.pi / 4, +1.0),
        (np.pi / 2, np.pi / 4, +1.0),
        (np.pi / 2, -np.pi / 4, -1.0),
    ]

    return sum(
        sign * exact_correlation(state, theta_a, theta_b)
        for theta_a, theta_b, sign in settings
    )


def main() -> None:
    shots = 20_000
    circuit, state = bell_state()

    # A0=Z、A1=X；B0=(Z+X)/sqrt(2)、B1=(Z-X)/sqrt(2)。
    settings = [
        ("E(A0,B0)", 0.0, np.pi / 4, +1.0),
        ("E(A0,B1)", 0.0, -np.pi / 4, +1.0),
        ("E(A1,B0)", np.pi / 2, np.pi / 4, +1.0),
        ("E(A1,B1)", np.pi / 2, -np.pi / 4, -1.0),
    ]

    print("Bell-state circuit:")
    print(circuit.draw(output="text"))
    print("Measurement settings:")
    print("A0=Z, A1=X")
    print("B0=(Z+X)/sqrt(2), B1=(Z-X)/sqrt(2)")

    exact_s = 0.0
    sampled_s = 0.0

    print("\nCorrelation       exact       sampled")
    print("--------------------------------------")

    for seed, (name, theta_a, theta_b, sign) in enumerate(
        settings,
        start=1200,
    ):
        exact = exact_correlation(state, theta_a, theta_b)
        sampled, _ = sampled_correlation(
            state,
            theta_a,
            theta_b,
            shots,
            seed,
        )

        exact_s += sign * exact
        sampled_s += sign * sampled
        print(f"{name:12s}   {exact: .6f}    {sampled: .6f}")

    # |00> 是 product state；相同量測設定不會違反 CHSH bound。
    product_state = Statevector.from_label("00")

    print("\nCHSH values:")
    print(f"Bell state, exact:       {exact_s:.6f}")
    print(f"Bell state, {shots} shots: {sampled_s:.6f}")
    print(f"Product state |00>:      {chsh_value(product_state):.6f}")
    print(f"Classical local bound:   {2.0:.6f}")
    print(f"Quantum Tsirelson bound: {2.0 * np.sqrt(2.0):.6f}")


if __name__ == "__main__":
    main()
