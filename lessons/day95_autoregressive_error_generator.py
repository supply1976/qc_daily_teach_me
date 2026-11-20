"""Day 95: Learn an autoregressive proposal for correlated logical failures."""

import numpy as np

from day91_repetition_code_distance_scaling import (
    exact_logical_error_rate,
    recovered_fidelity,
)
from day94_mixture_importance_sampling import (
    burst_masks,
    burst_probabilities,
    iid_pattern_probabilities,
    sample_proposal,
)


def masks_to_bits(masks, distance):
    positions = np.arange(distance, dtype=np.uint64)
    return (
        (masks[:, None] >> positions) & np.uint64(1)
    ).astype(np.int8)


def target_probabilities(
    masks,
    weights,
    distance,
    target_error_rate,
    target_burst_rate,
    allowed_bursts,
):
    return (
        (1.0 - target_burst_rate)
        * iid_pattern_probabilities(
            weights,
            distance,
            target_error_rate,
        )
        + target_burst_rate
        * burst_probabilities(masks, allowed_bursts)
    )


def mixture_proposal_probabilities(
    masks,
    weights,
    distance,
    proposal_error_rate,
    iid_mixture_weight,
    allowed_bursts,
):
    return (
        iid_mixture_weight
        * iid_pattern_probabilities(
            weights,
            distance,
            proposal_error_rate,
        )
        + (1.0 - iid_mixture_weight)
        * burst_probabilities(masks, allowed_bursts)
    )


def fit_weighted_markov(bits, weights, minimum_probability=1e-3):
    """Weighted MLE for P(x0=1), P(1|0), and P(1|1)."""
    total = np.sum(weights)
    initial_one = np.sum(weights * bits[:, 0]) / total

    previous = bits[:, :-1]
    current = bits[:, 1:]
    expanded_weights = weights[:, None]

    previous_zero = previous == 0
    previous_one = previous == 1
    zero_denominator = np.sum(expanded_weights * previous_zero)
    one_denominator = np.sum(expanded_weights * previous_one)

    one_given_zero = np.sum(
        expanded_weights * previous_zero * current
    ) / zero_denominator
    one_given_one = np.sum(
        expanded_weights * previous_one * current
    ) / one_denominator

    parameters = np.array([
        initial_one,
        one_given_zero,
        one_given_one,
    ])
    return np.clip(
        parameters,
        minimum_probability,
        1.0 - minimum_probability,
    )


def markov_log_probabilities(bits, parameters):
    initial_one, one_given_zero, one_given_one = parameters
    initial = bits[:, 0]
    log_probability = (
        initial * np.log(initial_one)
        + (1 - initial) * np.log(1.0 - initial_one)
    )

    previous = bits[:, :-1]
    current = bits[:, 1:]
    conditional_one = np.where(
        previous == 0,
        one_given_zero,
        one_given_one,
    )
    log_probability += np.sum(
        current * np.log(conditional_one)
        + (1 - current) * np.log(1.0 - conditional_one),
        axis=1,
    )
    return log_probability


def sample_markov(distance, parameters, trials, rng):
    initial_one, one_given_zero, one_given_one = parameters
    bits = np.zeros((trials, distance), dtype=np.int8)
    bits[:, 0] = rng.random(trials) < initial_one

    for position in range(1, distance):
        probability_one = np.where(
            bits[:, position - 1] == 0,
            one_given_zero,
            one_given_one,
        )
        bits[:, position] = rng.random(trials) < probability_one

    powers = np.left_shift(
        np.uint64(1),
        np.arange(distance, dtype=np.uint64),
    )
    masks = bits.astype(np.uint64) @ powers
    weights = np.sum(bits, axis=1).astype(int)
    probabilities = np.exp(markov_log_probabilities(bits, parameters))
    return masks, bits, weights, probabilities


def evaluate_proposal(
    masks,
    weights,
    proposal_probabilities,
    distance,
    target_error_rate,
    target_burst_rate,
    allowed_bursts,
):
    target = target_probabilities(
        masks,
        weights,
        distance,
        target_error_rate,
        target_burst_rate,
        allowed_bursts,
    )
    failures = weights > distance // 2
    contributions = failures * target / proposal_probabilities
    estimate = float(np.mean(contributions))
    standard_error = float(
        np.std(contributions, ddof=1) / np.sqrt(len(masks))
    )
    nonzero = contributions[failures]
    ess = float(np.sum(nonzero) ** 2 / np.sum(nonzero**2))
    burst_fraction = float(np.mean(
        burst_probabilities(masks, allowed_bursts) > 0.0
    ))
    return estimate, standard_error, ess, burst_fraction


def main():
    distance = 21
    target_error_rate = 0.02
    target_burst_rate = 6.0e-14
    burst_length = distance // 2 + 1
    allowed_bursts = burst_masks(distance, burst_length)
    pilot_trials = 300_000
    evaluation_trials = 300_000
    rng = np.random.default_rng(9501)
    exact_total = (
        (1.0 - target_burst_rate)
        * exact_logical_error_rate(distance, target_error_rate)
        + target_burst_rate
    )

    # A broad defensive mixture supplies weighted examples of both modes.
    pilot_masks, pilot_weights, _, _ = sample_proposal(
        distance=distance,
        proposal_error_rate=0.5246,
        iid_mixture_weight=0.50,
        allowed_bursts=allowed_bursts,
        trials=pilot_trials,
        rng=rng,
    )
    pilot_target = target_probabilities(
        pilot_masks,
        pilot_weights,
        distance,
        target_error_rate,
        target_burst_rate,
        allowed_bursts,
    )
    pilot_proposal = mixture_proposal_probabilities(
        pilot_masks,
        pilot_weights,
        distance,
        proposal_error_rate=0.5246,
        iid_mixture_weight=0.50,
        allowed_bursts=allowed_bursts,
    )
    pilot_failures = pilot_weights > distance // 2
    training_weights = pilot_failures * pilot_target / pilot_proposal
    pilot_bits = masks_to_bits(pilot_masks, distance)

    markov_parameters = fit_weighted_markov(
        pilot_bits,
        training_weights,
    )
    iid_parameter = float(
        np.sum(training_weights * pilot_weights / distance)
        / np.sum(training_weights)
    )

    print("Autoregressive proposal for correlated logical failures")
    print(f"pilot samples:          {pilot_trials:,}")
    print(f"exact logical error:    {exact_total:.12e}")
    print(f"weighted iid fit q:     {iid_parameter:.9f}")
    print("weighted Markov fit:")
    print(f"  P(x0=1):              {markov_parameters[0]:.9f}")
    print(f"  P(x_i=1 | x_(i-1)=0): {markov_parameters[1]:.9f}")
    print(f"  P(x_i=1 | x_(i-1)=1): {markov_parameters[2]:.9f}")
    print(
        "  persistence gap:      "
        f"{markov_parameters[2] - markov_parameters[1]:.9f}"
    )

    print("\nIndependent proposal evaluation")
    print("model          estimate          rel. SE      ESS     burst fraction")
    print("-" * 78)

    iid_bits = rng.random((evaluation_trials, distance)) < iid_parameter
    powers = np.left_shift(
        np.uint64(1),
        np.arange(distance, dtype=np.uint64),
    )
    iid_masks = iid_bits.astype(np.uint64) @ powers
    iid_weights = np.sum(iid_bits, axis=1).astype(int)
    iid_probabilities = iid_pattern_probabilities(
        iid_weights,
        distance,
        iid_parameter,
    )
    iid_result = evaluate_proposal(
        iid_masks,
        iid_weights,
        iid_probabilities,
        distance,
        target_error_rate,
        target_burst_rate,
        allowed_bursts,
    )

    markov_masks, _, markov_weights, markov_probabilities = sample_markov(
        distance,
        markov_parameters,
        evaluation_trials,
        rng,
    )
    markov_result = evaluate_proposal(
        markov_masks,
        markov_weights,
        markov_probabilities,
        distance,
        target_error_rate,
        target_burst_rate,
        allowed_bursts,
    )

    for label, result in (
        ("iid", iid_result),
        ("Markov", markov_result),
    ):
        estimate, standard_error, ess, burst_fraction = result
        print(
            f"{label:10s}"
            f"   {estimate:.12e}"
            f"   {standard_error / estimate:8.3%}"
            f"   {ess:8.1f}"
            f"   {burst_fraction:12.6%}"
        )

    print("\nQiskit correlation check")
    for length in (3, 4):
        mask = (1 << length) - 1
        _, fidelity = recovered_fidelity(7, mask)
        print(
            f"d=7, contiguous length={length}: "
            f"logical fidelity={fidelity:.9f}"
        )

    print("\nInterpretation")
    print(
        "P(1|1) > P(1|0) means the learned generator detects "
        "spatial error persistence without being told the burst locations."
    )
    print(
        "The Markov likelihood remains explicit, so every generated "
        "pattern can still receive an exact importance correction."
    )


if __name__ == "__main__":
    main()
