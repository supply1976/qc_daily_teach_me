"""Day 85: Bayesian filtering and smoothing for syndrome histories."""

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
    viterbi_decode,
)


def transition_matrix(data_error_probability):
    """P(current error state | previous error state)."""
    matrix = np.empty((NUM_STATES, NUM_STATES))
    for previous in range(NUM_STATES):
        for current in range(NUM_STATES):
            flips = int(previous ^ current).bit_count()
            matrix[previous, current] = (
                data_error_probability**flips
                * (1.0 - data_error_probability)
                ** (NUM_QUBITS - flips)
            )
    return matrix


def emission_vector(observation, measurement_error_probability):
    """P(observed syndrome | each possible hidden error state)."""
    disagreements = np.sum(
        SYNDROMES != observation[None, :],
        axis=1,
    )
    return (
        measurement_error_probability**disagreements
        * (1.0 - measurement_error_probability)
        ** (2 - disagreements)
    )


def normalize(probabilities):
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("Probability vector has zero mass.")
    return probabilities / total


def forward_filter(
    observations,
    data_error_probability,
    measurement_error_probability,
):
    """Compute P(x_t | y_1, ..., y_t) for every time step."""
    transition = transition_matrix(data_error_probability)
    posterior = np.zeros(NUM_STATES)
    posterior[0] = 1.0
    filtered = []

    for observation in observations:
        prediction = posterior @ transition
        likelihood = emission_vector(
            observation,
            measurement_error_probability,
        )
        posterior = normalize(prediction * likelihood)
        filtered.append(posterior.copy())

    return np.array(filtered)


def backward_smooth(
    observations,
    filtered,
    data_error_probability,
    measurement_error_probability,
):
    """Compute P(x_t | y_1, ..., y_T) using future observations."""
    transition = transition_matrix(data_error_probability)
    rounds = len(observations)
    backward = np.ones(NUM_STATES)
    smoothed = np.empty_like(filtered)
    smoothed[-1] = filtered[-1]

    for time in range(rounds - 2, -1, -1):
        next_likelihood = emission_vector(
            observations[time + 1],
            measurement_error_probability,
        )
        backward = transition @ (next_likelihood * backward)
        backward = normalize(backward)
        smoothed[time] = normalize(filtered[time] * backward)

    return smoothed


def posterior_decode(posteriors):
    return np.argmax(posteriors, axis=1).astype(np.int8)


def negative_log_probability(posteriors, hidden_path):
    probabilities = posteriors[
        np.arange(len(hidden_path)),
        hidden_path,
    ]
    return float(np.mean(-np.log(np.maximum(probabilities, 1e-300))))


def benchmark_inference(
    trials,
    rounds,
    data_error_probability,
    measurement_error_probability,
    seed,
):
    rng = np.random.default_rng(seed)
    correct = {
        "filter": 0,
        "smoother": 0,
        "Viterbi": 0,
    }
    final_correct = {
        "filter": 0,
        "smoother": 0,
        "Viterbi": 0,
    }
    nll = {"filter": 0.0, "smoother": 0.0}
    total_states = trials * rounds
    example = None

    for _ in range(trials):
        hidden, observed = generate_history(
            rounds,
            data_error_probability,
            measurement_error_probability,
            rng,
        )
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
        estimates = {
            "filter": posterior_decode(filtered),
            "smoother": posterior_decode(smoothed),
            "Viterbi": viterbi_decode(
                observed,
                data_error_probability,
                measurement_error_probability,
            ),
        }

        for name, estimate in estimates.items():
            correct[name] += int(np.sum(estimate == hidden))
            final_correct[name] += int(estimate[-1] == hidden[-1])

        nll["filter"] += negative_log_probability(filtered, hidden)
        nll["smoother"] += negative_log_probability(smoothed, hidden)

        if example is None:
            improvement = (
                (estimates["filter"] != hidden)
                & (estimates["smoother"] == hidden)
            )
            if np.any(improvement):
                example = (
                    hidden,
                    observed,
                    filtered,
                    smoothed,
                    estimates,
                )

    return {
        "state_accuracy": {
            name: value / total_states
            for name, value in correct.items()
        },
        "final_accuracy": {
            name: value / trials
            for name, value in final_correct.items()
        },
        "nll": {
            name: value / trials
            for name, value in nll.items()
        },
        "example": example,
    }


def main():
    rounds = 9
    data_error_probability = 0.015
    measurement_error_probability = 0.08
    trials = 5_000

    result = benchmark_inference(
        trials=trials,
        rounds=rounds,
        data_error_probability=data_error_probability,
        measurement_error_probability=measurement_error_probability,
        seed=8501,
    )

    print("Bayesian inference for repeated syndrome measurements")
    print(f"Trials:                       {trials}")
    print(f"Syndrome rounds:              {rounds}")
    print(f"Data flip per qubit/round:    {data_error_probability:.3f}")
    print(f"Measurement flip per bit:     {measurement_error_probability:.3f}")

    print("\nInference method     all-time accuracy    final accuracy")
    print("-" * 60)
    for name in ("filter", "smoother", "Viterbi"):
        print(
            f"{name:18s}"
            f"{result['state_accuracy'][name]:17.9f}"
            f"{result['final_accuracy'][name]:18.9f}"
        )

    print("\nMean negative log probability of the true hidden state")
    print(f"filter:   {result['nll']['filter']:.9f}")
    print(f"smoother: {result['nll']['smoother']:.9f}")

    example = result["example"]
    if example is None:
        return

    hidden, observed, filtered, smoothed, estimates = example
    print("\nExample: future syndrome repairs an earlier inference")
    print(
        "round  hidden  observed  filter(conf)  smoother(conf)  Viterbi"
    )
    print("-" * 73)
    for time in range(rounds):
        filter_state = estimates["filter"][time]
        smooth_state = estimates["smoother"][time]
        filter_confidence = filtered[time, filter_state]
        smooth_confidence = smoothed[time, smooth_state]
        print(
            f"{time + 1:5d}   {bitstring(hidden[time])}       "
            f"{syndrome_string(observed[time])}       "
            f"{bitstring(filter_state)}({filter_confidence:.3f})      "
            f"{bitstring(smooth_state)}({smooth_confidence:.3f})      "
            f"{bitstring(estimates['Viterbi'][time])}"
        )

    # A logical |0_L> makes every wrong X-pattern correction orthogonal.
    target = encoded_state(theta=0.0, phase=0.0)
    repaired_times = np.flatnonzero(
        (estimates["filter"] != hidden)
        & (estimates["smoother"] == hidden)
    )
    time = int(repaired_times[0])
    filter_fidelity = recovery_fidelity(
        target,
        int(hidden[time]),
        int(estimates["filter"][time]),
    )
    smooth_fidelity = recovery_fidelity(
        target,
        int(hidden[time]),
        int(estimates["smoother"][time]),
    )
    print("\nQiskit recovery check at the repaired round")
    print(f"round:              {time + 1}")
    print(f"filter fidelity:    {filter_fidelity:.9f}")
    print(f"smoother fidelity:  {smooth_fidelity:.9f}")

    final_difference = np.max(np.abs(filtered[-1] - smoothed[-1]))
    print("\nFinal-time identity")
    print(
        "max |filter(T) - smoother(T)| = "
        f"{final_difference:.3e}"
    )


if __name__ == "__main__":
    main()
