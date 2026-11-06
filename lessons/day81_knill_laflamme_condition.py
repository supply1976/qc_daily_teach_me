"""Day 81: Verify the Knill-Laflamme quantum error-correction condition."""

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import Pauli

from day80_five_qubit_perfect_code import (
    NUM_QUBITS,
    logical_codewords,
    single_qubit_pauli,
)


@dataclass(frozen=True)
class ErrorOperator:
    name: str
    matrix: np.ndarray


def identity_error():
    dimension = 2**NUM_QUBITS
    return ErrorOperator(
        name="I",
        matrix=np.eye(dimension, dtype=complex),
    )


def single_error(pauli, qubit):
    return ErrorOperator(
        name=f"{pauli}{qubit}",
        matrix=single_qubit_pauli(pauli, qubit).to_matrix(),
    )


def product_error(name, factors):
    """Build a product of Pauli factors acting on distinct qubits."""
    matrix = np.eye(2**NUM_QUBITS, dtype=complex)
    for pauli, qubit in factors:
        matrix = single_qubit_pauli(pauli, qubit).to_matrix() @ matrix
    return ErrorOperator(name=name, matrix=matrix)


def correctable_single_error_set():
    errors = [identity_error()]
    for qubit in range(NUM_QUBITS):
        for pauli in ("X", "Y", "Z"):
            errors.append(single_error(pauli, qubit))
    return errors


def logical_basis_matrix():
    """Return a 32x2 isometry whose columns are |0_L> and |1_L>."""
    logical_zero, logical_one = logical_codewords()
    return np.column_stack((logical_zero.data, logical_one.data))


def knill_laflamme_block(basis, first, second):
    """Compute <i_L| E_a^dagger E_b |j_L> for i,j in {0,1}."""
    return (
        basis.conj().T
        @ first.matrix.conj().T
        @ second.matrix
        @ basis
    )


def scalar_and_residual(block):
    """Separate a 2x2 block into c I and its non-correctable remainder."""
    scalar = np.trace(block) / 2.0
    residual = block - scalar * np.eye(2)
    return scalar, residual


def verify_error_set(errors):
    basis = logical_basis_matrix()
    coefficient_matrix = np.zeros(
        (len(errors), len(errors)),
        dtype=complex,
    )
    maximum_residual = 0.0
    worst_pair = None
    worst_block = None

    for row, first in enumerate(errors):
        for column, second in enumerate(errors):
            block = knill_laflamme_block(basis, first, second)
            scalar, residual = scalar_and_residual(block)
            coefficient_matrix[row, column] = scalar
            residual_norm = np.linalg.norm(residual, ord="fro")

            if residual_norm > maximum_residual:
                maximum_residual = residual_norm
                worst_pair = (first.name, second.name)
                worst_block = block

    return coefficient_matrix, maximum_residual, worst_pair, worst_block


def pauli_decomposition(block):
    """Decompose a logical 2x2 block into I, X, Y, Z coefficients."""
    paulis = {
        "I": Pauli("I").to_matrix(),
        "X": Pauli("X").to_matrix(),
        "Y": Pauli("Y").to_matrix(),
        "Z": Pauli("Z").to_matrix(),
    }
    return {
        name: 0.5 * np.trace(matrix.conj().T @ block)
        for name, matrix in paulis.items()
    }


def clean_complex(value, tolerance=1e-12):
    real = 0.0 if abs(value.real) < tolerance else value.real
    imag = 0.0 if abs(value.imag) < tolerance else value.imag
    return complex(real, imag)


def main():
    correctable = correctable_single_error_set()
    coefficients, residual, pair, block = verify_error_set(correctable)

    diagonal = np.real(np.diag(coefficients))
    off_diagonal = coefficients - np.diag(np.diag(coefficients))

    print("Knill-Laflamme condition for the [[5,1,3]] code")
    print(f"Correctable error operators: {len(correctable)}")
    print(f"KL blocks checked: {len(correctable) ** 2}")
    print(f"Maximum KL residual: {residual:.3e}")
    print(f"Worst numerical pair: {pair}")
    print(
        "Coefficient-matrix diagonal range: "
        f"[{diagonal.min():.9f}, {diagonal.max():.9f}]"
    )
    print(
        "Maximum off-diagonal coefficient: "
        f"{np.max(np.abs(off_diagonal)):.3e}"
    )
    print(
        "Coefficient matrix is identity: "
        f"{np.allclose(coefficients, np.eye(len(correctable)))}"
    )

    print("\nRepresentative correctable blocks")
    basis = logical_basis_matrix()
    lookup = {error.name: error for error in correctable}
    for first_name, second_name in (
        ("X0", "X0"),
        ("X0", "Y0"),
        ("Z3", "Z3"),
    ):
        example = knill_laflamme_block(
            basis,
            lookup[first_name],
            lookup[second_name],
        )
        print(f"<{first_name}^dagger {second_name}>_code =")
        print(np.array2string(example, precision=3, suppress_small=True))

    double_error = product_error(
        "X0X1",
        (("X", 0), ("X", 1)),
    )
    enlarged = correctable + [double_error]
    _, bad_residual, bad_pair, bad_block = verify_error_set(enlarged)

    print("\nAdd one uncorrectable two-qubit error")
    print(f"Added error: {double_error.name}")
    print(f"Maximum KL residual: {bad_residual:.9f}")
    print(f"Violating pair: {bad_pair}")
    print("Logical block for the violating pair:")
    print(np.array2string(bad_block, precision=3, suppress_small=True))

    decomposition = pauli_decomposition(bad_block)
    print("Logical Pauli decomposition:")
    for name, value in decomposition.items():
        print(f"  {name}: {clean_complex(value)}")

    print("\nInterpretation")
    print(
        "The correctable errors act as orthogonal syndrome labels and "
        "reveal no logical-state information."
    )
    print(
        "The two-qubit error overlaps a single-error syndrome while "
        "acting nontrivially on the logical qubit."
    )


if __name__ == "__main__":
    main()

