"""Day 52: SPAM bias and gauge freedom in gate-set tomography."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


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


def rotation_circuit(axis, angle):
    circuit = QuantumCircuit(1)
    if axis == "x":
        circuit.rx(float(angle), 0)
    elif axis == "y":
        circuit.ry(float(angle), 0)
    elif axis == "z":
        circuit.rz(float(angle), 0)
    else:
        raise ValueError("axis must be x, y, or z")
    return circuit


def bloch_rotation(circuit):
    """Convert a Qiskit unitary circuit to its 3x3 Bloch rotation."""
    unitary = Operator(circuit).data
    transfer = np.zeros((3, 3), dtype=float)

    for row, output_pauli in enumerate(PAULIS):
        for column, input_pauli in enumerate(PAULIS):
            mapped = unitary @ input_pauli @ unitary.conj().T
            transfer[row, column] = 0.5 * np.real(
                np.trace(output_pauli @ mapped)
            )
    return transfer


def sample_output_bloch(exact_bloch, shots, rng):
    estimates = []
    for expectation in exact_bloch:
        probability_plus = np.clip(
            0.5 * (1.0 + expectation),
            0.0,
            1.0,
        )
        plus_counts = rng.binomial(shots, probability_plus)
        estimates.append((2.0 * plus_counts - shots) / shots)
    return np.asarray(estimates)


def tomography(observed_map, shots, rng):
    outputs = []
    for input_bloch in PAULI_INPUTS:
        exact_output = observed_map @ input_bloch
        outputs.append(
            sample_output_bloch(exact_output, shots, rng)
        )

    # outputs = inputs @ coefficients, so map = coefficients.T.
    coefficients, *_ = np.linalg.lstsq(
        PAULI_INPUTS,
        np.asarray(outputs),
        rcond=None,
    )
    return coefficients.T


def nearest_rotation(matrix):
    """Polar projection of a noisy 3x3 matrix onto SO(3)."""
    left, _, right = np.linalg.svd(matrix)
    orientation = np.linalg.det(left @ right)
    return left @ np.diag([1.0, 1.0, orientation]) @ right


def rotation_angle_axis(rotation):
    cosine = np.clip(
        0.5 * (np.trace(rotation) - 1.0),
        -1.0,
        1.0,
    )
    angle = float(np.arccos(cosine))

    if abs(np.sin(angle)) < 1e-10:
        return angle, np.array([np.nan, np.nan, np.nan])

    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / (2.0 * np.sin(angle))
    axis /= np.linalg.norm(axis)
    return angle, axis


def average_unitary_fidelity(estimate, target):
    relative = target.T @ estimate
    return float((3.0 + np.trace(relative)) / 6.0)


def matrix_error(estimate, target):
    return float(np.linalg.norm(estimate - target))


def main():
    target_angle = 0.65
    preparation_angle = 0.18
    measurement_angle = -0.14
    shots = 4096
    rng = np.random.default_rng(20261008)

    target = bloch_rotation(
        rotation_circuit("x", target_angle)
    )
    preparation = bloch_rotation(
        rotation_circuit("z", preparation_angle)
    )
    measurement = bloch_rotation(
        rotation_circuit("y", measurement_angle)
    )

    # Standard process tomography assumes preparation=measurement=I.
    reference_observed = measurement @ preparation
    target_observed = measurement @ target @ preparation

    reference_estimate = tomography(
        reference_observed,
        shots,
        rng,
    )
    target_estimate = tomography(
        target_observed,
        shots,
        rng,
    )

    naive_rotation = nearest_rotation(target_estimate)

    # Two self-consistent gauges. Neither requires assigning the reference
    # error uniquely to state preparation or measurement.
    measurement_gauge = nearest_rotation(
        target_estimate @ np.linalg.inv(reference_estimate)
    )
    preparation_gauge = nearest_rotation(
        np.linalg.inv(reference_estimate) @ target_estimate
    )

    naive_angle, naive_axis = rotation_angle_axis(naive_rotation)
    measurement_gauge_angle, measurement_gauge_axis = (
        rotation_angle_axis(measurement_gauge)
    )
    preparation_gauge_angle, preparation_gauge_axis = (
        rotation_angle_axis(preparation_gauge)
    )

    true_axis = np.array([1.0, 0.0, 0.0])
    expected_measurement_axis = measurement @ true_axis
    expected_preparation_axis = preparation.T @ true_axis

    np.set_printoptions(precision=6, suppress=True)
    print("SPAM bias and gate-set tomography gauge freedom")
    print(f"Target gate:             Rx({target_angle:.6f})")
    print(f"Preparation error:      Rz({preparation_angle:.6f})")
    print(f"Measurement-frame error: Ry({measurement_angle:.6f})")
    print(f"Shots per Pauli setting: {shots}")

    print("\nExact maps seen by the experiment:")
    print("reference = M G =")
    print(reference_observed)
    print("target experiment = M T G =")
    print(target_observed)

    print("\nRotation estimates after polar projection:")
    print(
        " estimate                angle(rad)"
        "        axis [x, y, z]"
    )
    print("-" * 75)
    print(
        f" true target             {target_angle:.9f}"
        f"    {np.array2string(true_axis, precision=6)}"
    )
    print(
        f" naive process tomo      {naive_angle:.9f}"
        f"    {np.array2string(naive_axis, precision=6)}"
    )
    print(
        f" reference-normalized M  {measurement_gauge_angle:.9f}"
        f"    {np.array2string(measurement_gauge_axis, precision=6)}"
    )
    print(
        f" reference-normalized G  {preparation_gauge_angle:.9f}"
        f"    {np.array2string(preparation_gauge_axis, precision=6)}"
    )

    print("\nGauge predictions for the rotation axis:")
    print(
        "M gauge expected axis: "
        f"{np.array2string(expected_measurement_axis, precision=6)}"
    )
    print(
        "G gauge expected axis: "
        f"{np.array2string(expected_preparation_axis, precision=6)}"
    )

    print("\nErrors and invariants:")
    print(
        "naive fidelity versus target:       "
        f"{average_unitary_fidelity(naive_rotation, target):.9f}"
    )
    print(
        "naive matrix error:                 "
        f"{matrix_error(naive_rotation, target):.9f}"
    )
    print(
        "M-gauge angle error:                "
        f"{measurement_gauge_angle - target_angle:+.3e}"
    )
    print(
        "G-gauge angle error:                "
        f"{preparation_gauge_angle - target_angle:+.3e}"
    )
    print(
        "M-gauge axis error:                 "
        f"{matrix_error(measurement_gauge_axis, expected_measurement_axis):.6f}"
    )
    print(
        "G-gauge axis error:                 "
        f"{matrix_error(preparation_gauge_axis, expected_preparation_axis):.6f}"
    )


if __name__ == "__main__":
    main()
