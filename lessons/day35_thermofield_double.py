"""Day 35: purify a Gibbs thermal state as a thermofield double."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    entropy,
    partial_trace,
    state_fidelity,
)


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def hamiltonian(z_field, x_field):
    """Return H = z_field Z + x_field X."""
    return z_field * Z + x_field * X


def bell_state():
    """Prepare (|00> + |11>) / sqrt(2)."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return Statevector.from_instruction(circuit)


def imaginary_time_filter(matrix, beta):
    """Return exp(-beta H / 2), up to an irrelevant scalar factor."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Subtracting the ground energy avoids numerical overflow.
    factors = np.exp(
        -0.5 * beta * (eigenvalues - eigenvalues[0])
    )
    return (
        eigenvectors
        @ np.diag(factors)
        @ eigenvectors.conj().T
    )


def thermofield_double(matrix, beta):
    """Apply the imaginary-time filter to q0 of a Bell pair."""
    initial = bell_state().data
    local_filter = imaginary_time_filter(matrix, beta)

    # Qiskit orders two-qubit amplitudes as |q1 q0>, so an operation
    # on q0 is I(q1) tensor local_filter(q0).
    filtered = np.kron(I, local_filter) @ initial
    filtered /= np.linalg.norm(filtered)
    return Statevector(filtered)


def gibbs_state(matrix, beta):
    """Compute rho_beta = exp(-beta H) / Z by diagonalization."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    weights = np.exp(-beta * (eigenvalues - eigenvalues[0]))
    probabilities = weights / np.sum(weights)
    density = (
        eigenvectors
        @ np.diag(probabilities)
        @ eigenvectors.conj().T
    )
    return DensityMatrix(density)


def energy(density, matrix):
    return float(np.real(np.trace(density.data @ matrix)))


def density_purity(density):
    squared = density.data @ density.data
    return float(np.real(np.trace(squared)))


def ground_probability(density, matrix):
    _, eigenvectors = np.linalg.eigh(matrix)
    ground = eigenvectors[:, 0]
    return float(np.real(np.vdot(ground, density.data @ ground)))


def reduced_system_state(tfd):
    # Trace out auxiliary q1, leaving system q0.
    return partial_trace(tfd, [1])


def main():
    z_field = 0.7
    x_field = 1.1
    matrix = hamiltonian(z_field, x_field)
    omega = np.sqrt(z_field**2 + x_field**2)

    print(f"Hamiltonian H = {z_field} Z + {x_field} X")
    print(f"Energy scale omega: {omega:.9f}")
    print()
    print(
        " beta     T=1/beta       energy       analytic E  "
        " ground prob.   entropy(bit)    purity       fidelity"
    )
    print("-" * 105)

    maximum_fidelity_error = 0.0
    for beta in [0.0, 0.5, 1.0, 2.0, 4.0]:
        tfd = thermofield_double(matrix, beta)
        reduced = reduced_system_state(tfd)
        target = gibbs_state(matrix, beta)

        measured_energy = energy(reduced, matrix)
        analytic_energy = -omega * np.tanh(beta * omega)
        fidelity = state_fidelity(reduced, target)
        maximum_fidelity_error = max(
            maximum_fidelity_error,
            abs(1.0 - fidelity),
        )

        temperature = "infinity" if beta == 0 else f"{1 / beta:.3f}"
        print(
            f"{beta:5.1f}  {temperature:>11}  "
            f"{measured_energy:+.9f}  {analytic_energy:+.9f}  "
            f"{ground_probability(reduced, matrix):.9f}  "
            f"{entropy(reduced, base=2):.9f}  "
            f"{density_purity(reduced):.9f}  "
            f"{fidelity:.12f}"
        )

    print()
    print(
        "Maximum infidelity between partial trace and Gibbs state: "
        f"{maximum_fidelity_error:.3e}"
    )

    beta = 1.0
    reduced = reduced_system_state(thermofield_double(matrix, beta))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    energy_basis_probabilities = np.real(
        np.diag(eigenvectors.conj().T @ reduced.data @ eigenvectors)
    )

    print("\nReduced system density matrix at beta=1:")
    print(np.array2string(reduced.data, precision=9, suppress_small=True))
    print(
        "Computational-basis probabilities: "
        f"{np.real(np.diag(reduced.data))}"
    )
    print(
        "Energy-basis probabilities [ground, excited]: "
        f"{energy_basis_probabilities}"
    )
    print(
        "TFD entanglement entropy = Gibbs thermal entropy: "
        f"{entropy(reduced, base=2):.9f} bit"
    )


if __name__ == "__main__":
    main()
