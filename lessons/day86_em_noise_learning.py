"""Day 86: Learn quantum-error and readout rates with EM."""

import numpy as np

from day84_spacetime_viterbi_decoder import (
    NUM_QUBITS,
    NUM_STATES,
    SYNDROMES,
    bitstring,
    encoded_state,
    generate_history,
    recovery_fidelity,
    syndrome_string,
)
from day85_bayesian_syndrome_smoothing import (
    backward_smooth,
    forward_filter,
    posterior_decode,
    transition_matrix,
)


HAMMING_DISTANCE = np.array([
    [int(previous ^ current).bit_count()
     for current in range(NUM_STATES)]
    for previous in range(NUM_STATES)
], dtype=float)


def emission_table(observations, measurement_error_probability):
    """Return P(y_t | state) for every round and hidden state."""
    disagreements = np.sum(
        observations[:, None, :] != SYNDROMES[None, :, :],
        axis=2,
    )
    return (
        measurement_error_probability**disagreements
        * (1.0 - measurement_error_probability)
        ** (2 - disagreements)
    )


def forward_backward_statistics(
    observations,
    data_error_probability,
    measurement_error_probability,
):
    """E-step: expected bit flips and readout disagreements."""
    transition = transition_matrix(data_error_probability)
    emissions = emission_table(
        observations,
        measurement_error_probability,
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

    gamma = alpha * beta
    gamma /= np.sum(gamma, axis=1, keepdims=True)

    expected_data_flips = 0.0
    previous = np.zeros(NUM_STATES)
    previous[0] = 1.0
    for time in range(rounds):
        if time > 0:
            previous = alpha[time - 1]
        xi = (
            previous[:, None]
            * transition
            * emissions[time][None, :]
            * beta[time][None, :]
        )
        xi /= np.sum(xi)
        expected_data_flips += float(np.sum(xi * HAMMING_DISTANCE))

    disagreements = np.sum(
        observations[:, None, :] != SYNDROMES[None, :, :],
        axis=2,
    )
    expected_readout_flips = float(np.sum(gamma * disagreements))
    log_likelihood = float(np.sum(np.log(scales)))

    return (
        expected_data_flips,
        expected_readout_flips,
        log_likelihood,
    )


def fit_noise_rates(
    observation_sequences,
    initial_data_error,
    initial_measurement_error,
    iterations,
):
    """Baum-Welch/EM updates for the two physical noise parameters."""
    data_error = float(initial_data_error)
    measurement_error = float(initial_measurement_error)
    total_rounds = sum(len(sequence) for sequence in observation_sequences)
    history = []

    for iteration in range(iterations + 1):
        expected_data_flips = 0.0
        expected_readout_flips = 0.0
        log_likelihood = 0.0

        for observations in observation_sequences:
            data_flips, readout_flips, log_prob = (
                forward_backward_statistics(
                    observations,
                    data_error,
                    measurement_error,
                )
            )
            expected_data_flips += data_flips
            expected_readout_flips += readout_flips
            log_likelihood += log_prob

        history.append((
            iteration,
            data_error,
            measurement_error,
            log_likelihood,
        ))
        if iteration == iterations:
            break

        # Expected Bernoulli failures / number of opportunities.
        data_error = expected_data_flips / (
            NUM_QUBITS * total_rounds
        )
        measurement_error = expected_readout_flips / (
            2 * total_rounds
        )

        # Avoid log(0) in a later E-step.
        data_error = float(np.clip(data_error, 1e-8, 0.499999))
        measurement_error = float(np.clip(
            measurement_error,
            1e-8,
            0.499999,
        ))

    return data_error, measurement_error, history


def make_dataset(
    sequence_count,
    rounds,
    data_error_probability,
    measurement_error_probability,
    seed,
):
    rng = np.random.default_rng(seed)
    hidden_sequences = []
    observation_sequences = []
    for _ in range(sequence_count):
        hidden, observed = generate_history(
            rounds,
            data_error_probability,
            measurement_error_probability,
            rng,
        )
        hidden_sequences.append(hidden)
        observation_sequences.append(observed)
    return hidden_sequences, observation_sequences


def decode_accuracy(
    hidden_sequences,
    observation_sequences,
    data_error_probability,
    measurement_error_probability,
):
    correct = 0
    total = 0
    estimates = []
    for hidden, observed in zip(
        hidden_sequences,
        observation_sequences,
    ):
        filtered = forward_filter(
            observed,
            data_error_probability,
            measurement_error_probability,
        )
        smoothed = backward_smooth(
            observed,
            filtered,
            data_error_probability,
            measurement_error_probability,
        )
        estimate = posterior_decode(smoothed)
        correct += int(np.sum(estimate == hidden))
        total += len(hidden)
        estimates.append(estimate)
    return correct / total, estimates


def main():
    true_data_error = 0.020
    true_measurement_error = 0.120
    training_sequences = 400
    training_rounds = 25

    _, train_observations = make_dataset(
        training_sequences,
        training_rounds,
        true_data_error,
        true_measurement_error,
        seed=8601,
    )

    initial_data_error = 0.080
    initial_measurement_error = 0.250
    learned_data, learned_measurement, history = fit_noise_rates(
        train_observations,
        initial_data_error,
        initial_measurement_error,
        iterations=20,
    )

    print("EM learning of quantum data and syndrome readout noise")
    print(f"Training sequences: {training_sequences}")
    print(f"Rounds per sequence: {training_rounds}")
    print("\niteration    data p       readout q      log likelihood")
    print("-" * 61)
    checkpoints = {0, 1, 2, 5, 10, 15, 20}
    for iteration, data_p, readout_q, log_prob in history:
        if iteration in checkpoints:
            print(
                f"{iteration:9d}    {data_p:.9f}   "
                f"{readout_q:.9f}   {log_prob:.6f}"
            )

    print("\nParameter recovery")
    print("                    true          initial       learned")
    print("-" * 62)
    print(
        f"data error p      {true_data_error:.9f}    "
        f"{initial_data_error:.9f}   {learned_data:.9f}"
    )
    print(
        f"readout error q   {true_measurement_error:.9f}    "
        f"{initial_measurement_error:.9f}   "
        f"{learned_measurement:.9f}"
    )

    test_hidden, test_observations = make_dataset(
        sequence_count=2_000,
        rounds=12,
        data_error_probability=true_data_error,
        measurement_error_probability=true_measurement_error,
        seed=8602,
    )
    settings = {
        "miscalibrated": (
            initial_data_error,
            initial_measurement_error,
        ),
        "EM learned": (learned_data, learned_measurement),
        "oracle": (true_data_error, true_measurement_error),
    }

    print("\nSmoothed hidden-state decoding on held-out histories")
    print("decoder model       state accuracy")
    print("-" * 42)
    decoded = {}
    for name, (data_p, readout_q) in settings.items():
        accuracy, estimates = decode_accuracy(
            test_hidden,
            test_observations,
            data_p,
            readout_q,
        )
        decoded[name] = estimates
        print(f"{name:18s}  {accuracy:.9f}")

    example = None
    for sequence_index, hidden in enumerate(test_hidden):
        wrong = decoded["miscalibrated"][sequence_index] != hidden
        repaired = decoded["EM learned"][sequence_index] == hidden
        positions = np.flatnonzero(wrong & repaired)
        if len(positions) > 0:
            example = (sequence_index, int(positions[0]))
            break

    if example is not None:
        sequence_index, time = example
        hidden = test_hidden[sequence_index]
        observed = test_observations[sequence_index]
        print("\nExample repaired by calibration")
        print(f"round:              {time + 1}")
        print(f"observed syndrome:  {syndrome_string(observed[time])}")
        print(f"true error:         {bitstring(hidden[time])}")
        for name in settings:
            print(
                f"{name + ' estimate:':20s}"
                f"{bitstring(decoded[name][sequence_index][time])}"
            )

        logical_zero = encoded_state(theta=0.0, phase=0.0)
        bad_fidelity = recovery_fidelity(
            logical_zero,
            int(hidden[time]),
            int(decoded["miscalibrated"][sequence_index][time]),
        )
        learned_fidelity = recovery_fidelity(
            logical_zero,
            int(hidden[time]),
            int(decoded["EM learned"][sequence_index][time]),
        )
        print("\nQiskit logical-|0> recovery fidelity")
        print(f"miscalibrated: {bad_fidelity:.9f}")
        print(f"EM learned:    {learned_fidelity:.9f}")


if __name__ == "__main__":
    main()
