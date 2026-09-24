"""Day 38: non-commuting transverse-field quantum Boltzmann machine."""

import numpy as np
from qiskit.quantum_info import (
    DensityMatrix,
    SparsePauliOp,
    entropy,
    state_fidelity,
)


def matrix_exponential_hermitian(matrix, scale):
    """Compute exp(scale * matrix) for a Hermitian matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (
        eigenvectors
        @ np.diag(np.exp(scale * eigenvalues))
        @ eigenvectors.conj().T
    )


def hamiltonian_parts(coupling, transverse_field):
    """Return A=-J ZZ and B=-Gamma(X0+X1)."""
    classical = SparsePauliOp.from_list([
        ("ZZ", -float(coupling)),
    ]).to_matrix()
    transverse = SparsePauliOp.from_list([
        ("IX", -float(transverse_field)),
        ("XI", -float(transverse_field)),
    ]).to_matrix()
    return classical, transverse


def gibbs_density(matrix, beta):
    exponential = matrix_exponential_hermitian(matrix, -beta)
    return DensityMatrix(exponential / np.trace(exponential))


def symmetric_trotter_gibbs(classical, transverse, beta, steps):
    """Second-order product formula for exp[-beta(A+B)]."""
    classical_half = matrix_exponential_hermitian(
        classical,
        -beta / (2 * steps),
    )
    transverse_full = matrix_exponential_hermitian(
        transverse,
        -beta / steps,
    )
    one_step = classical_half @ transverse_full @ classical_half
    approximation = np.linalg.matrix_power(one_step, steps)

    # Remove floating-point anti-Hermitian residue before normalization.
    approximation = 0.5 * (
        approximation + approximation.conj().T
    )
    return DensityMatrix(approximation / np.trace(approximation))


def expectation(density, pauli_label):
    operator = SparsePauliOp.from_list([(pauli_label, 1.0)])
    return float(np.real(density.expectation_value(operator)))


def l1_coherence(density):
    diagonal = np.diag(np.diag(density.data))
    return float(np.sum(np.abs(density.data - diagonal)))


def trace_distance(first, second):
    singular_values = np.linalg.svd(
        first.data - second.data,
        compute_uv=False,
    )
    return float(0.5 * np.sum(singular_values))


def summary(density):
    probabilities = np.real(np.diag(density.data))
    return {
        "probabilities": probabilities,
        "zz": expectation(density, "ZZ"),
        "x_average": 0.5 * (
            expectation(density, "IX")
            + expectation(density, "XI")
        ),
        "coherence": l1_coherence(density),
        "entropy": float(entropy(density, base=2)),
        "purity": float(np.real(np.trace(density.data @ density.data))),
    }


def sample_density(density, shots, seed):
    density.seed(seed)
    raw = density.sample_counts(shots)
    return {
        format(index, "02b"): int(raw.get(format(index, "02b"), 0))
        for index in range(4)
    }


def main():
    beta = 1.0
    coupling = 0.5 * np.log(9.0)
    transverse_field = 0.7

    classical, transverse = hamiltonian_parts(
        coupling,
        transverse_field,
    )
    classical_gibbs = gibbs_density(classical, beta)
    quantum_gibbs = gibbs_density(classical + transverse, beta)

    commutator = classical @ transverse - transverse @ classical
    print(f"J = {coupling:.9f}, Gamma = {transverse_field:.3f}, beta = {beta:.1f}")
    print(f"Frobenius norm ||[A,B]||: {np.linalg.norm(commutator):.9f}\n")

    for name, density in [
        ("Commuting model (Gamma=0)", classical_gibbs),
        ("Transverse-field model", quantum_gibbs),
    ]:
        values = summary(density)
        print(name)
        print("  p[00, 01, 10, 11] =", np.array2string(
            values["probabilities"], precision=9
        ))
        print(f"  <Z0 Z1>              = {values['zz']:+.9f}")
        print(f"  average <X>          = {values['x_average']:+.9f}")
        print(f"  l1 coherence         = {values['coherence']:.9f}")
        print(f"  entropy              = {values['entropy']:.9f} bit")
        print(f"  purity               = {values['purity']:.9f}\n")

    print("Symmetric imaginary-time Suzuki-Trotter approximation:")
    print(" steps      trace distance       infidelity")
    print("-" * 51)
    errors = []

    for steps in [1, 2, 4, 8, 16, 32]:
        approximation = symmetric_trotter_gibbs(
            classical,
            transverse,
            beta,
            steps,
        )
        distance = trace_distance(approximation, quantum_gibbs)
        infidelity = 1.0 - state_fidelity(
            approximation,
            quantum_gibbs,
        )
        errors.append(distance)
        print(f"{steps:6d}      {distance:.9e}      {infidelity:.9e}")

    print("\nTrace-distance ratios error(r) / error(2r):")
    print(np.array2string(
        np.array(errors[:-1]) / np.array(errors[1:]),
        precision=6,
    ))
    print(
        "\nGenerated computational-basis counts (10000 shots): "
        f"{sample_density(quantum_gibbs, 10000, seed=3800)}"
    )


if __name__ == "__main__":
    main()
