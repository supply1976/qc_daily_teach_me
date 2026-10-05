"""Day 49: channel unitarity distinguishes coherent and stochastic noise."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus, Operator, random_statevector


I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULIS = [X, Y, Z]


def apply_kraus(matrix, kraus_operators):
    output = np.zeros_like(matrix, dtype=complex)
    for operator in kraus_operators:
        output += operator @ matrix @ operator.conj().T
    return output


def coherent_rz_channel(angle):
    circuit = QuantumCircuit(1)
    circuit.rz(float(angle), 0)
    return [Operator(circuit).data]


def depolarizing_channel(alpha):
    """Return Pauli Kraus operators with Bloch shrinkage alpha."""
    probability_identity = (1.0 + 3.0 * alpha) / 4.0
    probability_pauli = (1.0 - alpha) / 4.0

    return [
        np.sqrt(probability_identity) * I2,
        np.sqrt(probability_pauli) * X,
        np.sqrt(probability_pauli) * Y,
        np.sqrt(probability_pauli) * Z,
    ]


def amplitude_damping_channel(gamma):
    return [
        np.array([
            [1.0, 0.0],
            [0.0, np.sqrt(1.0 - gamma)],
        ], dtype=complex),
        np.array([
            [0.0, np.sqrt(gamma)],
            [0.0, 0.0],
        ], dtype=complex),
    ]


def pauli_transfer_data(kraus_operators):
    """Return the 3x3 unital block T and nonunital translation t."""
    transfer = np.zeros((3, 3), dtype=float)

    for row, output_pauli in enumerate(PAULIS):
        for column, input_pauli in enumerate(PAULIS):
            mapped = apply_kraus(
                input_pauli,
                kraus_operators,
            )
            transfer[row, column] = 0.5 * np.real(
                np.trace(output_pauli @ mapped)
            )

    mapped_identity = apply_kraus(I2, kraus_operators)
    translation = np.array([
        0.5 * np.real(np.trace(pauli @ mapped_identity))
        for pauli in PAULIS
    ])
    return transfer, translation


def average_fidelity(transfer):
    alpha = np.trace(transfer) / 3.0
    return float(0.5 * (1.0 + alpha))


def unitarity(transfer):
    return float(
        np.trace(transfer.T @ transfer) / 3.0
    )


def sampled_unitarity(kraus_operators, samples, rng):
    """Estimate u with random antipodal pure-state pairs."""
    squared_lengths = []

    for _ in range(samples):
        seed = int(rng.integers(0, 2**31 - 1))
        state = random_statevector(2, seed=seed).data
        rho_plus = np.outer(state, state.conj())

        # For a pure qubit, I-rho is its antipodal orthogonal state.
        rho_minus = I2 - rho_plus
        output_difference = (
            apply_kraus(rho_plus, kraus_operators)
            - apply_kraus(rho_minus, kraus_operators)
        )

        transformed_bloch = np.array([
            0.5 * np.real(
                np.trace(pauli @ output_difference)
            )
            for pauli in PAULIS
        ])
        squared_lengths.append(
            transformed_bloch @ transformed_bloch
        )

    values = np.asarray(squared_lengths)
    mean = float(np.mean(values))
    standard_error = float(
        np.std(values, ddof=1) / np.sqrt(samples)
    )
    return mean, standard_error


def main():
    coherent_angle = 0.20
    samples = 10000
    rng = np.random.default_rng(20261005)

    # Match all three channels to the same Clifford-twirled RB alpha.
    matched_alpha = (
        1.0 + 2.0 * np.cos(coherent_angle)
    ) / 3.0

    # For amplitude damping, T=diag(s,s,s^2), so
    # (2s+s^2)/3 = matched_alpha and gamma=1-s^2.
    amplitude_factor = (
        -1.0 + np.sqrt(1.0 + 3.0 * matched_alpha)
    )
    damping_probability = 1.0 - amplitude_factor**2

    channels = {
        "coherent Rz": coherent_rz_channel(coherent_angle),
        "depolarizing": depolarizing_channel(matched_alpha),
        "amplitude damping": amplitude_damping_channel(
            damping_probability
        ),
    }

    print("Noise channels matched by average RB fidelity")
    print(f"Coherent Rz angle:          {coherent_angle:.9f}")
    print(f"Matched RB alpha:           {matched_alpha:.9f}")
    print(
        "Matched amplitude-damping gamma: "
        f"{damping_probability:.9f}"
    )
    print(f"Random antipodal pairs:     {samples}")

    print("\nChannel summary:")
    print(
        " channel                 CPTP    avg fidelity    "
        "unitarity       sampled u      std error    ||t||"
    )
    print("-" * 97)

    results = {}
    for name, operators in channels.items():
        transfer, translation = pauli_transfer_data(operators)
        exact_unitarity = unitarity(transfer)
        sampled, standard_error = sampled_unitarity(
            operators,
            samples,
            rng,
        )
        cptp = Kraus(operators).is_cptp()

        results[name] = (
            transfer,
            translation,
            average_fidelity(transfer),
            exact_unitarity,
            sampled,
            standard_error,
        )

        print(
            f" {name:21s} {str(cptp):5s}"
            f"    {results[name][2]:.9f}"
            f"    {exact_unitarity:.9f}"
            f"    {sampled:.9f}"
            f"    {standard_error:.2e}"
            f"    {np.linalg.norm(translation):.9f}"
        )

    print("\nPauli-transfer blocks T:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(np.array2string(result[0], precision=6))
        print(
            "translation t = "
            + np.array2string(result[1], precision=6)
        )

    print("\nKey comparison:")
    print(
        "  coherent and depolarizing fidelity difference: "
        f"{results['coherent Rz'][2] - results['depolarizing'][2]:+.3e}"
    )
    print(
        "  coherent minus depolarizing unitarity:         "
        f"{results['coherent Rz'][3] - results['depolarizing'][3]:+.9f}"
    )


if __name__ == "__main__":
    main()
