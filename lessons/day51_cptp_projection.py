"""Day 51: project a noisy process-tomography estimate onto CPTP maps."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus, Operator


I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULIS = [X, Y, Z]

PAULI_INPUTS = np.array([
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
])


def channel_kraus(angle, gamma):
    """Amplitude damping followed by a coherent Rz rotation."""
    circuit = QuantumCircuit(1)
    circuit.rz(float(angle), 0)
    rotation = Operator(circuit).data

    damping_0 = np.array([
        [1.0, 0.0],
        [0.0, np.sqrt(1.0 - gamma)],
    ], dtype=complex)
    damping_1 = np.array([
        [0.0, np.sqrt(gamma)],
        [0.0, 0.0],
    ], dtype=complex)
    return [rotation @ damping_0, rotation @ damping_1]


def density_from_bloch(vector):
    density = I2.copy()
    for coefficient, pauli in zip(vector, PAULIS):
        density += coefficient * pauli
    return 0.5 * density


def bloch_vector(density):
    return np.array([
        np.real(np.trace(pauli @ density))
        for pauli in PAULIS
    ])


def apply_kraus(density, operators):
    output = np.zeros_like(density, dtype=complex)
    for operator in operators:
        output += operator @ density @ operator.conj().T
    return output


def sample_bloch(density, shots, rng):
    sampled = []
    for expectation in bloch_vector(density):
        probability_plus = np.clip(
            0.5 * (1.0 + expectation),
            0.0,
            1.0,
        )
        plus_counts = rng.binomial(shots, probability_plus)
        sampled.append((2.0 * plus_counts - shots) / shots)
    return np.asarray(sampled)


def linear_inversion(operators, shots, rng):
    outputs = []
    for input_bloch in PAULI_INPUTS:
        output = apply_kraus(
            density_from_bloch(input_bloch),
            operators,
        )
        outputs.append(sample_bloch(output, shots, rng))

    design = np.column_stack([
        PAULI_INPUTS,
        np.ones(len(PAULI_INPUTS)),
    ])
    coefficients, *_ = np.linalg.lstsq(
        design,
        np.asarray(outputs),
        rcond=None,
    )
    return coefficients[:3].T, coefficients[3]


def analytic_affine_map(angle, gamma):
    contraction = np.diag([
        np.sqrt(1.0 - gamma),
        np.sqrt(1.0 - gamma),
        1.0 - gamma,
    ])
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rotation @ contraction, np.array([0.0, 0.0, gamma])


def choi_from_affine(transfer, translation):
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
    return np.block([
        [0.5 * (mapped_identity + mapped_z),
         0.5 * (mapped_x + 1.0j * mapped_y)],
        [0.5 * (mapped_x - 1.0j * mapped_y),
         0.5 * (mapped_identity - mapped_z)],
    ])


def affine_from_choi(choi):
    blocks = choi.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)
    mapped_00 = blocks[0, 0]
    mapped_01 = blocks[0, 1]
    mapped_10 = blocks[1, 0]
    mapped_11 = blocks[1, 1]

    mapped_identity = mapped_00 + mapped_11
    mapped_paulis = [
        mapped_01 + mapped_10,
        -1.0j * mapped_01 + 1.0j * mapped_10,
        mapped_00 - mapped_11,
    ]

    translation = np.array([
        0.5 * np.real(np.trace(pauli @ mapped_identity))
        for pauli in PAULIS
    ])
    transfer = np.zeros((3, 3), dtype=float)
    for row, output_pauli in enumerate(PAULIS):
        for column, mapped in enumerate(mapped_paulis):
            transfer[row, column] = 0.5 * np.real(
                np.trace(output_pauli @ mapped)
            )
    return transfer, translation


def partial_trace_output(choi):
    tensor = choi.reshape(2, 2, 2, 2)
    return np.trace(tensor, axis1=1, axis2=3)


def project_positive_semidefinite(matrix):
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    return (
        vectors
        @ np.diag(np.clip(values, 0.0, None))
        @ vectors.conj().T
    )


def project_trace_preserving(choi):
    residual = partial_trace_output(choi) - I2
    projected = choi - 0.5 * np.kron(residual, I2)
    return 0.5 * (projected + projected.conj().T)


def project_cptp_dykstra(choi, tolerance=1e-12, max_iterations=10000):
    """Euclidean projection onto PSD and TP sets using Dykstra's method."""
    estimate = 0.5 * (choi + choi.conj().T)
    positive_correction = np.zeros_like(estimate)
    trace_correction = np.zeros_like(estimate)

    for iteration in range(1, max_iterations + 1):
        positive_input = estimate + positive_correction
        positive = project_positive_semidefinite(positive_input)
        positive_correction = positive_input - positive

        trace_input = positive + trace_correction
        updated = project_trace_preserving(trace_input)
        trace_correction = trace_input - updated

        change = np.linalg.norm(updated - estimate)
        estimate = updated

        minimum_eigenvalue = np.min(np.linalg.eigvalsh(estimate))
        trace_residual = np.linalg.norm(
            partial_trace_output(estimate) - I2
        )
        if (
            change < tolerance
            and minimum_eigenvalue >= -10.0 * tolerance
            and trace_residual < 10.0 * tolerance
        ):
            return estimate, iteration

    raise RuntimeError("CPTP projection did not converge")


def channel_diagnostics(name, choi, true_choi):
    eigenvalues = np.linalg.eigvalsh(choi)
    trace_residual = np.linalg.norm(
        partial_trace_output(choi) - I2
    )
    true_error = np.linalg.norm(choi - true_choi)
    print(
        f" {name:13s}  {np.min(eigenvalues):+.9f}"
        f"       {trace_residual:.3e}"
        f"           {true_error:.9f}"
    )


def validation_rms(transfer, translation, operators, samples, rng):
    squared_errors = []
    for _ in range(samples):
        vector = rng.normal(size=3)
        vector /= np.linalg.norm(vector)
        exact_output = bloch_vector(
            apply_kraus(density_from_bloch(vector), operators)
        )
        prediction = transfer @ vector + translation
        squared_errors.append(np.sum((prediction - exact_output) ** 2))
    return float(np.sqrt(np.mean(squared_errors)))


def main():
    angle = 0.25
    gamma = 0.12
    shots = 2048
    validation_samples = 1000
    rng = np.random.default_rng(20261007)

    operators = channel_kraus(angle, gamma)
    assert Kraus(operators).is_cptp()

    true_transfer, true_translation = analytic_affine_map(angle, gamma)
    raw_transfer, raw_translation = linear_inversion(
        operators,
        shots,
        rng,
    )

    true_choi = choi_from_affine(true_transfer, true_translation)
    raw_choi = choi_from_affine(raw_transfer, raw_translation)
    psd_only_choi = project_positive_semidefinite(raw_choi)
    cptp_choi, iterations = project_cptp_dykstra(raw_choi)

    cptp_transfer, cptp_translation = affine_from_choi(cptp_choi)

    raw_rms = validation_rms(
        raw_transfer,
        raw_translation,
        operators,
        validation_samples,
        np.random.default_rng(11),
    )
    cptp_rms = validation_rms(
        cptp_transfer,
        cptp_translation,
        operators,
        validation_samples,
        np.random.default_rng(11),
    )

    np.set_printoptions(precision=6, suppress=True)
    print("CPTP-constrained quantum process tomography")
    print(f"Rz angle:                {angle:.6f}")
    print(f"Amplitude damping gamma: {gamma:.6f}")
    print(f"Shots per Pauli setting: {shots}")
    print(f"Dykstra iterations:      {iterations}")

    print("\nPhysicality diagnostics:")
    print(
        " estimate       minimum Choi eigenvalue"
        "    ||Tr_out(J)-I||_F    ||J-J_true||_F"
    )
    print("-" * 82)
    channel_diagnostics("raw LS", raw_choi, true_choi)
    channel_diagnostics("PSD only", psd_only_choi, true_choi)
    channel_diagnostics("CPTP", cptp_choi, true_choi)

    print("\nRaw affine estimate:")
    print("T_raw =")
    print(raw_transfer)
    print(f"t_raw = {raw_translation}")

    print("\nCPTP-projected affine estimate:")
    print("T_CPTP =")
    print(cptp_transfer)
    print(f"t_CPTP = {cptp_translation}")

    print("\nHeld-out prediction:")
    print(f"raw Bloch RMS error:   {raw_rms:.9f}")
    print(f"CPTP Bloch RMS error:  {cptp_rms:.9f}")
    print(
        "RMS improvement:        "
        f"{100.0 * (1.0 - cptp_rms / raw_rms):.3f}%"
    )


if __name__ == "__main__":
    main()
