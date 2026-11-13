"""Day 88: Decode correlated two-qubit burst errors."""

import numpy as np

from day84_spacetime_viterbi_decoder import (
    NUM_QUBITS,
    NUM_STATES,
    SYNDROMES,
    bitstring,
    encoded_state,
    recovery_fidelity,
    syndrome_string,
)


BURST_MASK = 0b011  # Correlated X0 X1 event.


def independent_delta_distribution(single_flip_probability):
    """Probability of every three-bit error increment."""
    probabilities = np.empty(NUM_STATES)
    for delta in range(NUM_STATES):
        weight = int(delta).bit_count()
        probabilities[delta] = (
            single_flip_probability**weight
            * (1.0 - single_flip_probability)
            ** (NUM_QUBITS - weight)
        )
    return probabilities


def correlated_delta_distribution(
    single_flip_probability,
    burst_probability,
):
    """Convolve independent flips with an optional X0 X1 burst."""
    independent = independent_delta_distribution(
        single_flip_probability
    )
    probabilities = np.empty(NUM_STATES)
    for delta in range(NUM_STATES):
        probabilities[delta] = (
            (1.0 - burst_probability) * independent[delta]
            + burst_probability * independent[delta ^ BURST_MASK]
        )
    return probabilities


def transition_matrix(single_flip_probability, burst_probability):
    delta_probabilities = correlated_delta_distribution(
        single_flip_probability,
        burst_probability,
    )
    matrix = np.empty((NUM_STATES, NUM_STATES))
    for previous in range(NUM_STATES):
        for current in range(NUM_STATES):
            matrix[previous, current] = delta_probabilities[
                previous ^ current
            ]
    return matrix


def emission_table(observations, readout_error_probability):
    disagreements = np.sum(
        observations[:, None, :] != SYNDROMES[None, :, :],
        axis=2,
    )
    return (
        readout_error_probability**disagreements
        * (1.0 - readout_error_probability)
        ** (2 - disagreements)
    )


def sequence_log_likelihood(
    observations,
    transition,
    readout_error_probability,
):
    emissions = emission_table(
        observations,
        readout_error_probability,
    )
    posterior = np.zeros(NUM_STATES)
    posterior[0] = 1.0
    log_likelihood = 0.0
    for likelihood in emissions:
        unnormalized = (posterior @ transition) * likelihood
        scale = float(np.sum(unnormalized))
        posterior = unnormalized / scale
        log_likelihood += np.log(scale)
    return log_likelihood


def dataset_log_likelihood(
    observation_sequences,
    single_flip_probability,
    burst_probability,
    readout_error_probability,
):
    transition = transition_matrix(
        single_flip_probability,
        burst_probability,
    )
    return sum(
        sequence_log_likelihood(
            observations,
            transition,
            readout_error_probability,
        )
        for observations in observation_sequences
    )


def grid_fit_noise_model(
    observation_sequences,
    single_flip_grid,
    burst_grid,
    readout_error_probability,
):
    best = None
    surface = []
    for single_flip in single_flip_grid:
        for burst in burst_grid:
            log_likelihood = dataset_log_likelihood(
                observation_sequences,
                single_flip,
                burst,
                readout_error_probability,
            )
            candidate = (log_likelihood, single_flip, burst)
            surface.append(candidate)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best, surface


def smooth_states(
    observations,
    transition,
    readout_error_probability,
):
    emissions = emission_table(
        observations,
        readout_error_probability,
    )
    rounds = len(observations)
    alpha = np.empty((rounds, NUM_STATES))
    scales = np.empty(rounds)
    previous = np.zeros(NUM_STATES)
    previous[0] = 1.0

    for time in range(rounds):
        unnormalized = (previous @ transition) * emissions[time]
        scales[time] = np.sum(unnormalized)
        alpha[time] = unnormalized / scales[time]
        previous = alpha[time]

    beta = np.ones((rounds, NUM_STATES))
    for time in range(rounds - 2, -1, -1):
        beta[time] = transition @ (
            emissions[time + 1] * beta[time + 1]
        )
        beta[time] /= scales[time + 1]

    posterior = alpha * beta
    posterior /= np.sum(posterior, axis=1, keepdims=True)
    return posterior


def generate_correlated_history(
    rounds,
    single_flip_probability,
    burst_probability,
    readout_error_probability,
    rng,
):
    error_state = 0
    hidden = []
    observations = []
    burst_events = []
    for _ in range(rounds):
        independent_mask = 0
        for qubit in range(NUM_QUBITS):
            if rng.random() < single_flip_probability:
                independent_mask ^= 1 << qubit
        burst = rng.random() < burst_probability
        delta = independent_mask ^ (BURST_MASK if burst else 0)
        error_state ^= delta
        hidden.append(error_state)
        burst_events.append(burst)

        readout_flips = (
            rng.random(2) < readout_error_probability
        ).astype(np.int8)
        observations.append(SYNDROMES[error_state] ^ readout_flips)

    return (
        np.array(hidden, dtype=np.int8),
        np.array(observations, dtype=np.int8),
        np.array(burst_events, dtype=bool),
    )


def make_dataset(
    sequence_count,
    rounds,
    single_flip_probability,
    burst_probability,
    readout_error_probability,
    seed,
):
    rng = np.random.default_rng(seed)
    hidden_sequences = []
    observation_sequences = []
    burst_sequences = []
    for _ in range(sequence_count):
        hidden, observed, bursts = generate_correlated_history(
            rounds,
            single_flip_probability,
            burst_probability,
            readout_error_probability,
            rng,
        )
        hidden_sequences.append(hidden)
        observation_sequences.append(observed)
        burst_sequences.append(bursts)
    return hidden_sequences, observation_sequences, burst_sequences


def decode_accuracy(
    hidden_sequences,
    observation_sequences,
    single_flip_probability,
    burst_probability,
    readout_error_probability,
):
    transition = transition_matrix(
        single_flip_probability,
        burst_probability,
    )
    correct = 0
    total = 0
    estimates = []
    for hidden, observed in zip(hidden_sequences, observation_sequences):
        posterior = smooth_states(
            observed,
            transition,
            readout_error_probability,
        )
        estimate = np.argmax(posterior, axis=1).astype(np.int8)
        correct += int(np.sum(estimate == hidden))
        total += len(hidden)
        estimates.append(estimate)
    return correct / total, estimates


def main():
    true_single_flip = 0.010
    true_burst = 0.030
    readout_error = 0.080

    _, train_observations, _ = make_dataset(
        sequence_count=300,
        rounds=25,
        single_flip_probability=true_single_flip,
        burst_probability=true_burst,
        readout_error_probability=readout_error,
        seed=8801,
    )

    single_grid = np.linspace(0.005, 0.025, 9)
    burst_grid = np.linspace(0.000, 0.060, 13)
    correlated_fit, _ = grid_fit_noise_model(
        train_observations,
        single_grid,
        burst_grid,
        readout_error,
    )
    independent_fit, _ = grid_fit_noise_model(
        train_observations,
        np.linspace(0.005, 0.060, 23),
        np.array([0.0]),
        readout_error,
    )

    correlated_log_likelihood, learned_single, learned_burst = (
        correlated_fit
    )
    independent_log_likelihood, independent_single, _ = independent_fit

    print("Correlated X0 X1 burst-noise model")
    print("\nSyndrome-only maximum-likelihood calibration")
    print("model                 single p     burst c      log likelihood")
    print("-" * 69)
    print(
        f"independent best      {independent_single:.6f}    "
        f"{0.0:.6f}    {independent_log_likelihood:.6f}"
    )
    print(
        f"correlated best       {learned_single:.6f}    "
        f"{learned_burst:.6f}    {correlated_log_likelihood:.6f}"
    )
    print(
        f"true parameters       {true_single_flip:.6f}    "
        f"{true_burst:.6f}"
    )
    print(
        "log-likelihood gain:  "
        f"{correlated_log_likelihood - independent_log_likelihood:.6f}"
    )

    independent_delta = correlated_delta_distribution(
        independent_single,
        0.0,
    )
    correlated_delta = correlated_delta_distribution(
        learned_single,
        learned_burst,
    )
    print("\nLearned probability of selected error increments")
    print("delta       independent       correlated")
    print("-" * 47)
    for delta in (0b000, 0b001, 0b010, 0b011, 0b100):
        print(
            f" {bitstring(delta)}       "
            f"{independent_delta[delta]:.9f}      "
            f"{correlated_delta[delta]:.9f}"
        )

    test_hidden, test_observations, test_bursts = make_dataset(
        sequence_count=2_000,
        rounds=15,
        single_flip_probability=true_single_flip,
        burst_probability=true_burst,
        readout_error_probability=readout_error,
        seed=8802,
    )
    models = {
        "independent fitted": (independent_single, 0.0),
        "correlated fitted": (learned_single, learned_burst),
        "correlated oracle": (true_single_flip, true_burst),
    }

    print("\nHeld-out smoothed-state accuracy")
    print("decoder model          accuracy")
    print("-" * 40)
    decoded = {}
    for name, (single_flip, burst) in models.items():
        accuracy, estimates = decode_accuracy(
            test_hidden,
            test_observations,
            single_flip,
            burst,
            readout_error,
        )
        decoded[name] = estimates
        print(f"{name:20s}  {accuracy:.9f}")

    example = None
    for sequence_index, hidden in enumerate(test_hidden):
        wrong = decoded["independent fitted"][sequence_index] != hidden
        repaired = decoded["correlated fitted"][sequence_index] == hidden
        burst = test_bursts[sequence_index]
        positions = np.flatnonzero(wrong & repaired & burst)
        if len(positions):
            example = sequence_index, int(positions[0])
            break

    if example is not None:
        sequence_index, time = example
        hidden = test_hidden[sequence_index]
        observed = test_observations[sequence_index]
        print("\nExample at a true X0 X1 burst event")
        print(f"round:             {time + 1}")
        print(f"observed syndrome: {syndrome_string(observed[time])}")
        print(f"true error state:  {bitstring(hidden[time])}")
        for name in models:
            print(
                f"{name + ':':22s}"
                f"{bitstring(decoded[name][sequence_index][time])}"
            )

        logical_zero = encoded_state(theta=0.0, phase=0.0)
        independent_fidelity = recovery_fidelity(
            logical_zero,
            int(hidden[time]),
            int(decoded["independent fitted"][sequence_index][time]),
        )
        correlated_fidelity = recovery_fidelity(
            logical_zero,
            int(hidden[time]),
            int(decoded["correlated fitted"][sequence_index][time]),
        )
        print("\nQiskit logical-|0> recovery fidelity")
        print(f"independent: {independent_fidelity:.9f}")
        print(f"correlated:  {correlated_fidelity:.9f}")


if __name__ == "__main__":
    main()
