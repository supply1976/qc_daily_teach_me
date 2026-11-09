"""Day 84: Space-time syndrome decoding with the Viterbi algorithm."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


NUM_QUBITS = 3
NUM_STATES = 2**NUM_QUBITS


def syndrome(error_state):
    """Return eigenvalue bits of Z0 Z1 and Z1 Z2."""
    x0 = (error_state >> 0) & 1
    x1 = (error_state >> 1) & 1
    x2 = (error_state >> 2) & 1
    return np.array([x0 ^ x1, x1 ^ x2], dtype=np.int8)


SYNDROMES = np.array([
    syndrome(state) for state in range(NUM_STATES)
])

# Minimum-weight correction for each two-bit syndrome.
MINIMUM_WEIGHT_DECODER = {
    (0, 0): 0b000,
    (1, 0): 0b001,
    (1, 1): 0b010,
    (0, 1): 0b100,
}


def encoded_state(theta=1.1, phase=0.7):
    circuit = QuantumCircuit(NUM_QUBITS)
    circuit.ry(theta, 0)
    circuit.rz(phase, 0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    return Statevector.from_instruction(circuit)


def apply_x_pattern(state, pattern):
    circuit = QuantumCircuit(NUM_QUBITS)
    for qubit in range(NUM_QUBITS):
        if (pattern >> qubit) & 1:
            circuit.x(qubit)
    return state.evolve(circuit)


def recovery_fidelity(target, true_error, inferred_error):
    damaged = apply_x_pattern(target, true_error)
    recovered = apply_x_pattern(damaged, inferred_error)
    return float(state_fidelity(target, recovered))


def hamming_weight(value):
    return int(value).bit_count()


def log_transition_matrix(data_error_probability):
    """Log P(x_t | x_{t-1}) for independent flips on three qubits."""
    matrix = np.empty((NUM_STATES, NUM_STATES))
    for previous in range(NUM_STATES):
        for current in range(NUM_STATES):
            flips = hamming_weight(previous ^ current)
            probability = (
                data_error_probability**flips
                * (1.0 - data_error_probability)
                ** (NUM_QUBITS - flips)
            )
            matrix[previous, current] = np.log(probability)
    return matrix


def log_emission_probability(
    observation,
    error_state,
    measurement_error_probability,
):
    """Log P(observed syndrome | hidden physical error state)."""
    disagreements = int(np.sum(
        observation != SYNDROMES[error_state]
    ))
    probability = (
        measurement_error_probability**disagreements
        * (1.0 - measurement_error_probability)
        ** (2 - disagreements)
    )
    return np.log(probability)


def viterbi_decode(
    observations,
    data_error_probability,
    measurement_error_probability,
):
    """Return the most probable hidden error path."""
    transition = log_transition_matrix(data_error_probability)

    # Before round one, the physical state is known to have no error.
    scores = np.full(NUM_STATES, -np.inf)
    scores[0] = 0.0
    backpointers = []

    for observation in observations:
        candidates = scores[:, None] + transition
        parents = np.argmax(candidates, axis=0)
        new_scores = candidates[parents, np.arange(NUM_STATES)]

        emissions = np.array([
            log_emission_probability(
                observation,
                state,
                measurement_error_probability,
            )
            for state in range(NUM_STATES)
        ])
        scores = new_scores + emissions
        backpointers.append(parents)

    final_state = int(np.argmax(scores))
    path = [final_state]
    for parents in reversed(backpointers[1:]):
        path.append(int(parents[path[-1]]))
    path.reverse()
    return np.array(path, dtype=np.int8)


def final_round_decode(observations):
    return MINIMUM_WEIGHT_DECODER[tuple(observations[-1])]


def static_majority_decode(observations):
    """The Day-83 decoder, intentionally misapplied to changing data."""
    voted = (
        np.sum(observations, axis=0) > len(observations) // 2
    ).astype(np.int8)
    return MINIMUM_WEIGHT_DECODER[tuple(voted)]


def generate_history(
    rounds,
    data_error_probability,
    measurement_error_probability,
    rng,
):
    error_state = 0
    hidden_path = []
    observations = []

    for _ in range(rounds):
        new_error = 0
        for qubit in range(NUM_QUBITS):
            if rng.random() < data_error_probability:
                new_error ^= 1 << qubit
        error_state ^= new_error
        hidden_path.append(error_state)

        observed = syndrome(error_state)
        readout_flips = (
            rng.random(2) < measurement_error_probability
        ).astype(np.int8)
        observations.append(observed ^ readout_flips)

    return (
        np.array(hidden_path, dtype=np.int8),
        np.array(observations, dtype=np.int8),
    )


def benchmark_decoders(
    trials,
    rounds,
    data_error_probability,
    measurement_error_probability,
    seed,
):
    rng = np.random.default_rng(seed)
    successes = {
        "final round": 0,
        "static majority": 0,
        "Viterbi": 0,
    }
    example = None

    for _ in range(trials):
        hidden, observed = generate_history(
            rounds,
            data_error_probability,
            measurement_error_probability,
            rng,
        )
        true_final = int(hidden[-1])
        estimates = {
            "final round": final_round_decode(observed),
            "static majority": static_majority_decode(observed),
            "Viterbi": int(viterbi_decode(
                observed,
                data_error_probability,
                measurement_error_probability,
            )[-1]),
        }

        for name, estimate in estimates.items():
            successes[name] += estimate == true_final

        if (
            example is None
            and estimates["Viterbi"] == true_final
            and estimates["static majority"] != true_final
        ):
            example = (hidden, observed, estimates)

    rates = {
        name: count / trials
        for name, count in successes.items()
    }
    return rates, example


def bitstring(state):
    """Display physical errors in q2 q1 q0 order."""
    return format(int(state), "03b")


def syndrome_string(bits):
    return "".join(str(int(bit)) for bit in bits)


def main():
    print("Three-qubit repetition code: space-time syndrome decoding")
    print("Hidden state = cumulative X-error pattern q2 q1 q0")

    print("\nSyndrome degeneracy")
    print("error    syndrome    logical complement    same syndrome")
    print("-" * 58)
    for state in range(4):
        complement = state ^ 0b111
        print(
            f" {bitstring(state)}       {syndrome_string(SYNDROMES[state])}"
            f"             {bitstring(complement)}"
            f"                  "
            f"{np.array_equal(SYNDROMES[state], SYNDROMES[complement])}"
        )

    target = encoded_state()
    print("\nQiskit check: residual logical X is not successful recovery")
    print(
        "true=111, inferred=000, fidelity="
        f"{recovery_fidelity(target, 0b111, 0b000):.9f}"
    )
    print(
        "true=111, inferred=111, fidelity="
        f"{recovery_fidelity(target, 0b111, 0b111):.9f}"
    )

    trials = 10_000
    rounds = 9
    data_error_probability = 0.015
    measurement_error_probability = 0.08
    rates, example = benchmark_decoders(
        trials=trials,
        rounds=rounds,
        data_error_probability=data_error_probability,
        measurement_error_probability=measurement_error_probability,
        seed=8401,
    )

    print("\nPhenomenological-noise benchmark")
    print(f"Trials:                       {trials}")
    print(f"Syndrome rounds:              {rounds}")
    print(f"Data flip per qubit/round:    {data_error_probability:.3f}")
    print(f"Measurement flip per bit:     {measurement_error_probability:.3f}")
    print("\nDecoder             exact-recovery probability")
    print("-" * 50)
    for name, rate in rates.items():
        print(f"{name:20s}{rate:.9f}")

    if example is not None:
        hidden, observed, estimates = example
        inferred_path = viterbi_decode(
            observed,
            data_error_probability,
            measurement_error_probability,
        )
        print("\nExample where time history matters")
        print("round   hidden error   true syndrome   observed   Viterbi")
        print("-" * 62)
        for index in range(rounds):
            print(
                f"{index + 1:5d}      {bitstring(hidden[index])}"
                f"            {syndrome_string(SYNDROMES[hidden[index]])}"
                f"          {syndrome_string(observed[index])}"
                f"        {bitstring(inferred_path[index])}"
            )
        print("Final estimates:")
        for name, estimate in estimates.items():
            print(f"  {name:20s} -> {bitstring(estimate)}")


if __name__ == "__main__":
    main()
