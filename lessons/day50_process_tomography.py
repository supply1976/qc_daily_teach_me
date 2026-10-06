"""Day 50: reconstruct a single-qubit noise channel by process tomography."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus, Operator, Statevector, random_statevector


I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULIS = [X, Y, Z]


def state_preparation_circuits():
    """Return six Pauli-eigenstate preparations as Qiskit circuits."""
    circuits = {}

    plus_x = QuantumCircuit(1)
    plus_x.h(0)
    circuits["+X"] = plus_x

    minus_x = QuantumCircuit(1)
    minus_x.x(0)
    minus_x.h(0)
    circuits["-X"] = minus_x

    plus_y = QuantumCircuit(1)
    plus_y.h(0)
    plus_y.s(0)
    circuits["+Y"] = plus_y

    minus_y = QuantumCircuit(1)
    minus_y.h(0)
    minus_y.sdg(0)
    circuits["-Y"] = minus_y

    circuits["+Z"] = QuantumCircuit(1)

    minus_z = QuantumCircuit(1)
    minus_z.x(0)
    circuits["-Z"] = minus_z

    return circuits


def rz_after_amplitude_damping(angle, gamma):
    """Kraus operators for amplitude damping followed by Rz(angle)."""
    rotation = QuantumCircuit(1)
    rotation.rz(float(angle), 0)
    unitary = Operator(rotation).data

    damping_0 = np.array([
        [1.0, 0.0],
        [0.0, np.sqrt(1.0 - gamma)],
    ], dtype=complex)
    damping_1 = np.array([
        [0.0, np.sqrt(gamma)],
        [0.0, 0.0],
    ], dtype=complex)

    return [unitary @ damping_0, unitary @ damping_1]


def apply_kraus(density, kraus_operators):
    output = np.zeros_like(density, dtype=complex)
    for operator in kraus_operators:
        output += operator @ density @ operator.conj().T
    return output


def bloch_vector(density):
    return np.array([
        np.real(np.trace(pauli @ density))
        for pauli in PAULIS
    ])


def sample_bloch_vector(density, shots, rng):
    """Sample independent X, Y, and Z measurement settings."""
    estimates = []
    for expectation in bloch_vector(density):
        probability_plus = np.clip(
            0.5 * (1.0 + expectation),
            0.0,
            1.0,
        )
        plus_counts = rng.binomial(shots, probability_plus)
        estimates.append((2.0 * plus_counts - shots) / shots)
    return np.asarray(estimates)


def collect_tomography_data(kraus_operators, shots, rng):
    labels = []
    inputs = []
    exact_outputs = []
    sampled_outputs = []

    for label, circuit in state_preparation_circuits().items():
        state = Statevector.from_instruction(circuit).data
        input_density = np.outer(state, state.conj())
        output_density = apply_kraus(
            input_density,
            kraus_operators,
        )

        labels.append(label)
        inputs.append(bloch_vector(input_density))
        exact_outputs.append(bloch_vector(output_density))
        sampled_outputs.append(
            sample_bloch_vector(output_density, shots, rng)
        )

    return (
        labels,
        np.asarray(inputs),
        np.asarray(exact_outputs),
        np.asarray(sampled_outputs),
    )


def reconstruct_affine_map(inputs, outputs):
    """Fit r_out = T r_in + t by linear least squares."""
    design = np.column_stack([
        inputs,
        np.ones(len(inputs)),
    ])
    coefficients, *_ = np.linalg.lstsq(
        design,
        outputs,
        rcond=None,
    )

    transfer = coefficients[:3, :].T
    translation = coefficients[3, :]
    return transfer, translation


def analytic_affine_map(angle, gamma):
    damping = np.diag([
        np.sqrt(1.0 - gamma),
        np.sqrt(1.0 - gamma),
        1.0 - gamma,
    ])
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rotation @ damping, np.array([0.0, 0.0, gamma])


def choi_from_affine_map(transfer, translation):
    """Construct the unnormalized Choi matrix of a TP affine qubit map."""
    mapped_identity = I2.copy()
    for coefficient, pauli in zip(translation, PAULIS):
        mapped_identity += coefficient * pauli

    mapped_paulis = []
    for column in range(3):
        mapped = np.zeros((2, 2), dtype=complex)
        for row, pauli in enumerate(PAULIS):
            mapped += transfer[row, column] * pauli
        mapped_paulis.append(mapped)

    mapped_x, mapped_y, mapped_z = mapped_paulis
    mapped_00 = 0.5 * (mapped_identity + mapped_z)
    mapped_11 = 0.5 * (mapped_identity - mapped_z)
    mapped_01 = 0.5 * (mapped_x + 1.0j * mapped_y)
    mapped_10 = 0.5 * (mapped_x - 1.0j * mapped_y)

    choi = np.block([
        [mapped_00, mapped_01],
        [mapped_10, mapped_11],
    ])
    return 0.5 * (choi + choi.conj().T)


def validation_error(transfer, translation, kraus_operators, samples, rng):
    errors = []
    for _ in range(samples):
        seed = int(rng.integers(0, 2**31 - 1))
        state = random_statevector(2, seed=seed).data
        density = np.outer(state, state.conj())
        input_bloch = bloch_vector(density)

        exact_output = bloch_vector(
            apply_kraus(density, kraus_operators)
        )
        predicted_output = transfer @ input_bloch + translation
        errors.append(np.linalg.norm(predicted_output - exact_output))

    errors = np.asarray(errors)
    return float(np.sqrt(np.mean(errors**2))), float(np.max(errors))


def main():
    coherent_angle = 0.25
    damping_probability = 0.12
    shots = 2048
    validation_samples = 1000
    rng = np.random.default_rng(20261006)

    kraus_operators = rz_after_amplitude_damping(
        coherent_angle,
        damping_probability,
    )
    assert Kraus(kraus_operators).is_cptp()

    labels, inputs, exact_outputs, sampled_outputs = (
        collect_tomography_data(
            kraus_operators,
            shots,
            rng,
        )
    )

    true_transfer, true_translation = analytic_affine_map(
        coherent_angle,
        damping_probability,
    )
    exact_transfer, exact_translation = reconstruct_affine_map(
        inputs,
        exact_outputs,
    )
    sampled_transfer, sampled_translation = reconstruct_affine_map(
        inputs,
        sampled_outputs,
    )

    exact_choi_values = np.linalg.eigvalsh(
        choi_from_affine_map(
            exact_transfer,
            exact_translation,
        )
    )
    sampled_choi_values = np.linalg.eigvalsh(
        choi_from_affine_map(
            sampled_transfer,
            sampled_translation,
        )
    )

    rms_error, maximum_error = validation_error(
        sampled_transfer,
        sampled_translation,
        kraus_operators,
        validation_samples,
        rng,
    )

    np.set_printoptions(precision=6, suppress=True)
    print("Single-qubit quantum process tomography")
    print(f"Coherent Rz angle:       {coherent_angle:.6f}")
    print(f"Amplitude damping gamma: {damping_probability:.6f}")
    print(f"Shots per Pauli setting: {shots}")
    print(f"Total circuit shots:     {len(labels) * 3 * shots}")

    print("\nTomography data (output Bloch vectors):")
    print(" input       exact [X, Y, Z]             sampled [X, Y, Z]")
    print("-" * 76)
    for label, exact, sampled in zip(
        labels,
        exact_outputs,
        sampled_outputs,
    ):
        print(
            f" {label:3s}   "
            f"{np.array2string(exact, precision=6):27s}  "
            f"{np.array2string(sampled, precision=6)}"
        )

    print("\nTrue affine map:")
    print("T =")
    print(true_transfer)
    print(f"t = {true_translation}")

    print("\nSampled reconstruction:")
    print("T_hat =")
    print(sampled_transfer)
    print(f"t_hat = {sampled_translation}")

    print("\nReconstruction errors:")
    print(
        "exact-data ||T_hat-T||_F:  "
        f"{np.linalg.norm(exact_transfer - true_transfer):.3e}"
    )
    print(
        "sampled ||T_hat-T||_F:     "
        f"{np.linalg.norm(sampled_transfer - true_transfer):.6f}"
    )
    print(
        "sampled ||t_hat-t||_2:     "
        f"{np.linalg.norm(sampled_translation - true_translation):.6f}"
    )
    print(f"held-out Bloch RMS error:  {rms_error:.6f}")
    print(f"held-out maximum error:    {maximum_error:.6f}")

    print("\nChoi eigenvalues (complete positivity check):")
    print(f"exact:   {exact_choi_values}")
    print(f"sampled: {sampled_choi_values}")
    print(
        "sampled minimum eigenvalue: "
        f"{np.min(sampled_choi_values):+.6e}"
    )


if __name__ == "__main__":
    main()
