"""Day 87: Learn heterogeneous qubit and check error rates with EM."""

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
from day86_em_noise_learning import (
    fit_noise_rates,
    make_dataset,
)


NUM_CHECKS = 2
STATE_BITS = np.array([
    [(state >> qubit) & 1 for qubit in range(NUM_QUBITS)]
    for state in range(NUM_STATES)
], dtype=np.int8)
FLIP_BITS = np.bitwise_xor(
    STATE_BITS[:, None, :],
    STATE_BITS[None, :, :],
)


def heterogeneous_transition(data_error_rates):
    """P(x_t | x_{t-1}) for separate rates p0, p1, p2."""
    rates = np.asarray(data_error_rates, dtype=float)
    probabilities = np.where(
        FLIP_BITS == 1,
        rates[None, None, :],
        1.0 - rates[None, None, :],
    )
    return np.prod(probabilities, axis=2)


def heterogeneous_emissions(observations, readout_error_rates):
    """P(y_t | x_t) for separate stabilizer-readout rates q0, q1."""
    rates = np.asarray(readout_error_rates, dtype=float)
    mismatch = (
        observations[:, None, :]
        != SYNDROMES[None, :, :]
    )
    probabilities = np.where(
        mismatch,
        rates[None, None, :],
        1.0 - rates[None, None, :],
    )
    return np.prod(probabilities, axis=2), mismatch


def forward_backward(
    observations,
    data_error_rates,
    readout_error_rates,
):
    """Return posterior marginals and EM sufficient statistics."""
    transition = heterogeneous_transition(data_error_rates)
    emissions, mismatch = heterogeneous_emissions(
        observations,
        readout_error_rates,
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

    expected_data_flips = np.zeros(NUM_QUBITS)
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
        expected_data_flips += np.sum(
            xi[:, :, None] * FLIP_BITS,
            axis=(0, 1),
        )

    expected_readout_flips = np.sum(
        gamma[:, :, None] * mismatch,
        axis=(0, 1),
    )
    log_likelihood = float(np.sum(np.log(scales)))
    return (
        gamma,
        expected_data_flips,
        expected_readout_flips,
        log_likelihood,
    )


def fit_heterogeneous_rates(
    observation_sequences,
    initial_data_rates,
    initial_readout_rates,
    iterations,
):
    data_rates = np.asarray(initial_data_rates, dtype=float).copy()
    readout_rates = np.asarray(
        initial_readout_rates,
        dtype=float,
    ).copy()
    total_rounds = sum(len(sequence) for sequence in observation_sequences)
    history = []

    for iteration in range(iterations + 1):
        data_counts = np.zeros(NUM_QUBITS)
        readout_counts = np.zeros(NUM_CHECKS)
        log_likelihood = 0.0
        for observations in observation_sequences:
            _, data_flips, readout_flips, log_prob = forward_backward(
                observations,
                data_rates,
                readout_rates,
            )
            data_counts += data_flips
            readout_counts += readout_flips
            log_likelihood += log_prob

        history.append((
            iteration,
            data_rates.copy(),
            readout_rates.copy(),
            log_likelihood,
        ))
        if iteration == iterations:
            break

        data_rates = np.clip(
            data_counts / total_rounds,
            1e-8,
            0.499999,
        )
        readout_rates = np.clip(
            readout_counts / total_rounds,
            1e-8,
            0.499999,
        )

    return data_rates, readout_rates, history


def generate_heterogeneous_history(
    rounds,
    data_error_rates,
    readout_error_rates,
    rng,
):
    error_state = 0
    hidden = []
    observed = []
    for _ in range(rounds):
        flips = rng.random(NUM_QUBITS) < data_error_rates
        for qubit, flip in enumerate(flips):
            if flip:
                error_state ^= 1 << qubit
        hidden.append(error_state)

        readout_flips = (
            rng.random(NUM_CHECKS) < readout_error_rates
        ).astype(np.int8)
        observed.append(SYNDROMES[error_state] ^ readout_flips)

    return (
        np.array(hidden, dtype=np.int8),
        np.array(observed, dtype=np.int8),
    )


def make_heterogeneous_dataset(
    sequence_count,
    rounds,
    data_error_rates,
    readout_error_rates,
    seed,
):
    rng = np.random.default_rng(seed)
    hidden_sequences = []
    observation_sequences = []
    for _ in range(sequence_count):
        hidden, observed = generate_heterogeneous_history(
            rounds,
            data_error_rates,
            readout_error_rates,
            rng,
        )
        hidden_sequences.append(hidden)
        observation_sequences.append(observed)
    return hidden_sequences, observation_sequences


def heterogeneous_decode_accuracy(
    hidden_sequences,
    observation_sequences,
    data_error_rates,
    readout_error_rates,
):
    correct = 0
    total = 0
    decoded = []
    for hidden, observed in zip(hidden_sequences, observation_sequences):
        posterior, _, _, _ = forward_backward(
            observed,
            data_error_rates,
            readout_error_rates,
        )
        estimate = np.argmax(posterior, axis=1).astype(np.int8)
        correct += int(np.sum(estimate == hidden))
        total += len(hidden)
        decoded.append(estimate)
    return correct / total, decoded


def main():
    true_data_rates = np.array([0.010, 0.030, 0.060])
    true_readout_rates = np.array([0.050, 0.180])
    initial_data_rates = np.full(NUM_QUBITS, 0.080)
    initial_readout_rates = np.full(NUM_CHECKS, 0.250)

    _, train_observations = make_heterogeneous_dataset(
        sequence_count=500,
        rounds=30,
        data_error_rates=true_data_rates,
        readout_error_rates=true_readout_rates,
        seed=8701,
    )
    learned_data, learned_readout, history = fit_heterogeneous_rates(
        train_observations,
        initial_data_rates,
        initial_readout_rates,
        iterations=20,
    )

    # Fit the restricted homogeneous model to the same observations.
    homogeneous_p, homogeneous_q, homogeneous_history = fit_noise_rates(
        train_observations,
        initial_data_error=0.080,
        initial_measurement_error=0.250,
        iterations=20,
    )

    print("Heterogeneous EM calibration from syndrome histories")
    print("\niteration   p0       p1       p2       q0       q1       log L")
    print("-" * 78)
    for iteration, data_rates, readout_rates, log_prob in history:
        if iteration in {0, 1, 2, 5, 10, 20}:
            values = np.concatenate((data_rates, readout_rates))
            print(
                f"{iteration:9d}  "
                + "  ".join(f"{value:.5f}" for value in values)
                + f"  {log_prob:.3f}"
            )

    print("\nRecovered hardware-dependent rates")
    print("parameter       true       learned")
    print("-" * 38)
    names = ("p0", "p1", "p2", "q0", "q1")
    true_values = np.concatenate((true_data_rates, true_readout_rates))
    learned_values = np.concatenate((learned_data, learned_readout))
    for name, true, learned in zip(names, true_values, learned_values):
        print(f"{name:8s}    {true:.9f}  {learned:.9f}")

    print("\nRestricted homogeneous model")
    print(f"shared data p:    {homogeneous_p:.9f}")
    print(f"shared readout q: {homogeneous_q:.9f}")
    print(
        "log likelihood gap (heterogeneous - homogeneous): "
        f"{history[-1][3] - homogeneous_history[-1][3]:.6f}"
    )

    test_hidden, test_observations = make_heterogeneous_dataset(
        sequence_count=2_000,
        rounds=15,
        data_error_rates=true_data_rates,
        readout_error_rates=true_readout_rates,
        seed=8702,
    )
    models = {
        "homogeneous EM": (
            np.full(NUM_QUBITS, homogeneous_p),
            np.full(NUM_CHECKS, homogeneous_q),
        ),
        "heterogeneous EM": (learned_data, learned_readout),
        "oracle": (true_data_rates, true_readout_rates),
    }

    print("\nHeld-out smoothed-state accuracy")
    print("decoder model          accuracy")
    print("-" * 40)
    decoded = {}
    for name, (data_rates, readout_rates) in models.items():
        accuracy, estimates = heterogeneous_decode_accuracy(
            test_hidden,
            test_observations,
            data_rates,
            readout_rates,
        )
        decoded[name] = estimates
        print(f"{name:20s}  {accuracy:.9f}")

    example = None
    for sequence_index, hidden in enumerate(test_hidden):
        wrong = decoded["homogeneous EM"][sequence_index] != hidden
        repaired = decoded["heterogeneous EM"][sequence_index] == hidden
        positions = np.flatnonzero(wrong & repaired)
        if len(positions):
            example = sequence_index, int(positions[0])
            break

    if example is not None:
        sequence_index, time = example
        hidden = test_hidden[sequence_index]
        observed = test_observations[sequence_index]
        print("\nExample where hardware-aware calibration changes recovery")
        print(f"round:             {time + 1}")
        print(f"observed syndrome: {syndrome_string(observed[time])}")
        print(f"true error:        {bitstring(hidden[time])}")
        for name in models:
            print(
                f"{name + ':':22s}"
                f"{bitstring(decoded[name][sequence_index][time])}"
            )

        logical_zero = encoded_state(theta=0.0, phase=0.0)
        homogeneous_fidelity = recovery_fidelity(
            logical_zero,
            int(hidden[time]),
            int(decoded["homogeneous EM"][sequence_index][time]),
        )
        heterogeneous_fidelity = recovery_fidelity(
            logical_zero,
            int(hidden[time]),
            int(decoded["heterogeneous EM"][sequence_index][time]),
        )
        print("\nQiskit logical-|0> recovery fidelity")
        print(f"homogeneous:   {homogeneous_fidelity:.9f}")
        print(f"heterogeneous: {heterogeneous_fidelity:.9f}")


if __name__ == "__main__":
    main()
