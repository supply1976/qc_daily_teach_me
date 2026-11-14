"""Day 89: Space-time detection events for repeated syndrome rounds."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


NUM_DATA_QUBITS = 3
NUM_CHECKS = 2


def syndrome(error_pattern):
    x0 = (error_pattern >> 0) & 1
    x1 = (error_pattern >> 1) & 1
    x2 = (error_pattern >> 2) & 1
    return np.array([x0 ^ x1, x1 ^ x2], dtype=np.int8)


def syndrome_extraction_circuit(error_pattern, theta=1.1, phase=0.7):
    """Extract Z0Z1 and Z1Z2 parities into two ancillas."""
    circuit = QuantumCircuit(5)

    # Encode alpha|000> + beta|111> on data qubits q0, q1, q2.
    circuit.ry(theta, 0)
    circuit.rz(phase, 0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)

    for qubit in range(NUM_DATA_QUBITS):
        if (error_pattern >> qubit) & 1:
            circuit.x(qubit)

    # Ancilla q3 stores q0 XOR q1; q4 stores q1 XOR q2.
    circuit.cx(0, 3)
    circuit.cx(1, 3)
    circuit.cx(1, 4)
    circuit.cx(2, 4)
    return circuit


def qiskit_syndrome(error_pattern):
    state = Statevector.from_instruction(
        syndrome_extraction_circuit(error_pattern)
    )
    probabilities = state.probabilities_dict(qargs=[3, 4])
    outcome = max(probabilities, key=probabilities.get)

    # Qiskit prints q4 q3, while our array is [check0, check1].
    return np.array([
        int(outcome[1]),
        int(outcome[0]),
    ], dtype=np.int8), probabilities[outcome]


def observed_history(
    rounds,
    data_event_round=None,
    data_error_mask=0,
    measurement_flips=None,
):
    """Generate ideal persistent data errors plus transient readout flips."""
    measurement_flips = measurement_flips or {}
    current_error = 0
    observations = []

    for time in range(rounds):
        if data_event_round is not None and time == data_event_round:
            current_error ^= data_error_mask

        observed = syndrome(current_error).copy()
        for check in measurement_flips.get(time, ()):
            observed[check] ^= 1
        observations.append(observed)

    return np.array(observations, dtype=np.int8)


def detection_matrix(observations, initial_syndrome=None):
    """d_t = observed syndrome at t XOR observed syndrome at t-1."""
    if initial_syndrome is None:
        initial_syndrome = np.zeros(NUM_CHECKS, dtype=np.int8)
    previous = np.vstack((initial_syndrome, observations[:-1]))
    return np.bitwise_xor(observations, previous)


def event_coordinates(detections):
    """Return (round, check) coordinates using one-based round labels."""
    coordinates = np.argwhere(detections == 1)
    return [(int(time + 1), int(check)) for time, check in coordinates]


def classify_isolated_pair(detections):
    """Classify the simplest two-event signatures."""
    events = event_coordinates(detections)
    if len(events) != 2:
        return "not an isolated two-event pattern"

    (time_a, check_a), (time_b, check_b) = events
    if time_a == time_b and {check_a, check_b} == {0, 1}:
        return "spatial pair: likely middle-qubit X1 data error"
    if check_a == check_b and abs(time_a - time_b) == 1:
        return f"temporal pair: likely readout error on check {check_a}"
    return "longer path or multiple faults: graph matching required"


def edge_weights(data_error_probability, readout_error_probability):
    """Negative log-odds weights used by a matching decoder."""
    data_weight = -np.log(
        data_error_probability / (1.0 - data_error_probability)
    )
    measurement_weight = -np.log(
        readout_error_probability
        / (1.0 - readout_error_probability)
    )
    return float(data_weight), float(measurement_weight)


def print_history(title, observations):
    detections = detection_matrix(observations)
    print(f"\n{title}")
    print("round    observed syndrome    detection event")
    print("-" * 50)
    for time, (observed, detected) in enumerate(
        zip(observations, detections),
        start=1,
    ):
        print(
            f"{time:5d}          {observed[0]}{observed[1]}"
            f"                 {detected[0]}{detected[1]}"
        )
    print(f"event coordinates: {event_coordinates(detections)}")
    print(f"interpretation:    {classify_isolated_pair(detections)}")
    print(
        "nonzero syndrome bits / detection events: "
        f"{int(np.sum(observations))} / {int(np.sum(detections))}"
    )


def main():
    print("Qiskit stabilizer-syndrome extraction")
    print("error pattern q2q1q0    analytic    ancilla    probability")
    print("-" * 66)
    for error_pattern in range(8):
        analytic = syndrome(error_pattern)
        measured, probability = qiskit_syndrome(error_pattern)
        print(
            f"        {error_pattern:03b}             "
            f"{analytic[0]}{analytic[1]}          "
            f"{measured[0]}{measured[1]}       {probability:.9f}"
        )

    rounds = 8
    middle_data_error = observed_history(
        rounds,
        data_event_round=2,
        data_error_mask=0b010,
    )
    check_zero_measurement_error = observed_history(
        rounds,
        measurement_flips={2: (0,)},
    )
    boundary_data_error = observed_history(
        rounds,
        data_event_round=2,
        data_error_mask=0b001,
    )

    print_history(
        "Persistent X1 data error beginning at round 3",
        middle_data_error,
    )
    print_history(
        "One-round measurement error on check 0 at round 3",
        check_zero_measurement_error,
    )
    print_history(
        "Boundary X0 data error beginning at round 3",
        boundary_data_error,
    )

    data_weight, measurement_weight = edge_weights(
        data_error_probability=0.02,
        readout_error_probability=0.08,
    )
    print("\nMatching-graph edge weights: -log(p / (1-p))")
    print(f"spatial data-error edge:  {data_weight:.9f}")
    print(f"temporal readout edge:    {measurement_weight:.9f}")
    print(
        "A full decoder pairs detection events through minimum-total-"
        "weight paths, including spatial and temporal boundaries."
    )


if __name__ == "__main__":
    main()
