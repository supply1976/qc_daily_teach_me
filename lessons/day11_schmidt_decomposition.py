"""第 11 課：Schmidt decomposition、SVD 與糾纏強度。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Statevector,
    concurrence,
    entropy,
    partial_trace,
    schmidt_decomposition,
)


def make_state(p: float) -> tuple[QuantumCircuit, Statevector]:
    """建立 sqrt(p)|00> + sqrt(1-p)|11>。"""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")

    # Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    theta = 2.0 * np.arccos(np.sqrt(p))

    circuit = QuantumCircuit(2)
    circuit.ry(theta, 0)
    circuit.cx(0, 1)

    return circuit, Statevector.from_instruction(circuit)


def binary_entropy(p: float) -> float:
    """計算 -p log2(p) - (1-p) log2(1-p)。"""
    probabilities = np.array([p, 1.0 - p])
    nonzero = probabilities[probabilities > 0.0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def main() -> None:
    p = 0.80
    circuit, state = make_state(p)

    # Qiskit 使用 |q1 q0> 的 little-endian 排列。
    # qargs=[0] 表示 B=q0，而 A=q1。
    terms = schmidt_decomposition(state, qargs=[0])

    print("Circuit:")
    print(circuit.draw(output="text"))
    print("Statevector in |q1 q0> order:")
    print(np.round(state.data, 6))

    print("\nQiskit Schmidt terms:")
    for index, (coefficient, vector_a, vector_b) in enumerate(terms):
        print(f"term {index}: lambda = {coefficient:.6f}")
        print("  |u>_A =", np.round(vector_a.data, 6))
        print("  |v>_B =", np.round(vector_b.data, 6))

    # 對 amplitude matrix 做普通 NumPy SVD。
    # rows 對應 q1，columns 對應 q0。
    amplitude_matrix = state.data.reshape(2, 2)
    _, singular_values, _ = np.linalg.svd(amplitude_matrix)

    schmidt_coefficients = np.array(
        [coefficient for coefficient, _, _ in terms]
    )
    schmidt_probabilities = schmidt_coefficients**2
    schmidt_rank = int(np.count_nonzero(schmidt_coefficients > 1e-12))

    # 驗證 Schmidt terms 能重建原 statevector。
    reconstructed = sum(
        coefficient * np.kron(vector_a.data, vector_b.data)
        for coefficient, vector_a, vector_b in terms
    )
    reconstruction_error = np.linalg.norm(reconstructed - state.data)

    reduced_q1 = partial_trace(state, [0])
    reduced_eigenvalues = np.linalg.eigvalsh(reduced_q1.data)[::-1]

    print("\nNumPy SVD singular values:")
    print(np.round(singular_values, 6))

    print("Schmidt probabilities lambda^2:")
    print(np.round(schmidt_probabilities, 6))

    print("Eigenvalues of reduced density matrix rho_A:")
    print(np.round(reduced_eigenvalues, 6))

    print(f"Schmidt rank: {schmidt_rank}")
    print(f"Reconstruction error: {reconstruction_error:.3e}")
    print(f"Entanglement entropy: {entropy(reduced_q1, base=2):.6f} bits")
    print(f"Binary-entropy formula: {binary_entropy(p):.6f} bits")
    print(f"Concurrence: {concurrence(state):.6f}")

    print("\nEntanglement as p changes:")
    print("   p      rank    entropy    concurrence")
    print("-----------------------------------------")

    for test_p in [0.50, 0.80, 0.95, 1.00]:
        _, test_state = make_state(test_p)
        test_terms = schmidt_decomposition(test_state, qargs=[0])
        test_rank = sum(term[0] > 1e-12 for term in test_terms)
        test_reduced = partial_trace(test_state, [0])

        print(
            f" {test_p:5.2f}      {test_rank}      "
            f"{entropy(test_reduced, base=2):.6f}      "
            f"{concurrence(test_state):.6f}"
        )


if __name__ == "__main__":
    main()
