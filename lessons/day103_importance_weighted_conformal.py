"""Day 103: Importance-weighted conformal decoding under noise drift."""

import numpy as np

from day91_repetition_code_distance_scaling import syndrome
from day94_mixture_importance_sampling import burst_masks, sample_proposal
from day96_noise_aware_ml_decoder import (
    error_probability,
    qiskit_recovery_fidelity,
)
from day102_conformal_decoder_sets import (
    build_decoder_cache,
    conformal_quantile,
    evaluate_sets,
    nonconformity_scores,
    prediction_set,
    true_label,
)


def density_ratios(
    error_masks,
    distance,
    source_parameters,
    target_parameters,
    allowed_bursts,
):
    """Compute w(e)=P_target(e)/P_source(e) from two noise generators."""
    source_iid, source_burst = source_parameters
    target_iid, target_burst = target_parameters
    weights = np.empty(len(error_masks), dtype=float)
    for index, raw_mask in enumerate(error_masks):
        error_mask = int(raw_mask)
        source_probability = error_probability(
            error_mask,
            distance,
            source_iid,
            source_burst,
            allowed_bursts,
        )
        target_probability = error_probability(
            error_mask,
            distance,
            target_iid,
            target_burst,
            allowed_bursts,
        )
        weights[index] = target_probability / source_probability
    return weights


def weighted_quantile(values, weights, quantile):
    """Left-continuous weighted empirical quantile."""
    order = np.argsort(values)
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    target_mass = quantile * cumulative[-1]
    index = int(np.searchsorted(cumulative, target_mass, side="left"))
    return float(ordered_values[index])


def exact_target_score_distribution(
    distance,
    cache,
    target_parameters,
    allowed_bursts,
):
    """Enumerate all physical errors for an oracle target quantile."""
    target_iid, target_burst = target_parameters
    scores = []
    probabilities = []
    for error_mask in range(1 << distance):
        record = cache[syndrome(error_mask, distance)]
        label = true_label(error_mask, record)
        scores.append(1.0 - record["posterior"][label])
        probabilities.append(error_probability(
            error_mask,
            distance,
            target_iid,
            target_burst,
            allowed_bursts,
        ))
    return np.array(scores), np.array(probabilities)


def main():
    distance = 7
    burst_length = 4
    source_parameters = (0.20, 0.05)
    target_parameters = (0.25, 0.10)
    calibration_size = 20_000
    test_size = 200_000
    rng = np.random.default_rng(10301)

    allowed_bursts_array = burst_masks(distance, burst_length)
    allowed_bursts = frozenset(
        int(mask) for mask in allowed_bursts_array
    )
    cache = build_decoder_cache(
        distance,
        source_parameters[0],
        source_parameters[1],
        allowed_bursts,
        logit_scale=1.0,
    )

    calibration_masks, _, _, _ = sample_proposal(
        distance,
        source_parameters[0],
        1.0 - source_parameters[1],
        allowed_bursts_array,
        calibration_size,
        rng,
    )
    test_masks, _, _, _ = sample_proposal(
        distance,
        target_parameters[0],
        1.0 - target_parameters[1],
        allowed_bursts_array,
        test_size,
        rng,
    )

    calibration_scores = nonconformity_scores(
        calibration_masks,
        distance,
        cache,
    )
    importance_weights = density_ratios(
        calibration_masks,
        distance,
        source_parameters,
        target_parameters,
        allowed_bursts,
    )
    exact_scores, exact_probabilities = exact_target_score_distribution(
        distance,
        cache,
        target_parameters,
        allowed_bursts,
    )

    effective_sample_size = (
        np.sum(importance_weights) ** 2
        / np.sum(importance_weights**2)
    )

    print("Importance-weighted conformal decoding under noise drift")
    print(
        "source noise (p, r):       "
        f"({source_parameters[0]:.3f}, {source_parameters[1]:.3f})"
    )
    print(
        "target noise (p, r):       "
        f"({target_parameters[0]:.3f}, {target_parameters[1]:.3f})"
    )
    print(f"calibration samples:         {calibration_size:,}")
    print(f"mean importance weight:      {np.mean(importance_weights):.9f}")
    print(f"maximum importance weight:   {np.max(importance_weights):.9f}")
    print(f"importance-weight ESS:       {effective_sample_size:.1f}")

    print(
        "\nalpha  target   ordinary q / coverage"
        "       weighted q / coverage       oracle q"
    )
    print("-" * 84)
    results = {}
    for alpha in (0.20, 0.10, 0.05, 0.01):
        ordinary_threshold, _ = conformal_quantile(
            calibration_scores,
            alpha,
        )
        weighted_threshold = weighted_quantile(
            calibration_scores,
            importance_weights,
            1.0 - alpha,
        )
        oracle_threshold = weighted_quantile(
            exact_scores,
            exact_probabilities,
            1.0 - alpha,
        )
        ordinary_metrics = evaluate_sets(
            test_masks,
            distance,
            cache,
            ordinary_threshold,
        )
        weighted_metrics = evaluate_sets(
            test_masks,
            distance,
            cache,
            weighted_threshold,
        )
        results[alpha] = (
            ordinary_threshold,
            weighted_threshold,
            ordinary_metrics,
            weighted_metrics,
        )
        print(
            f"{alpha:5.2f}  {1.0 - alpha:6.1%}"
            f"     {ordinary_threshold:8.6f} /"
            f" {ordinary_metrics['coverage']:8.3%}"
            f"       {weighted_threshold:8.6f} /"
            f" {weighted_metrics['coverage']:8.3%}"
            f"       {oracle_threshold:8.6f}"
        )

    alpha = 0.05
    ordinary_threshold, weighted_threshold, ordinary_metrics, weighted_metrics = (
        results[alpha]
    )
    print("\nSet efficiency at alpha=0.05")
    print("method       avg set size   singleton rate   singleton error")
    print("-" * 68)
    for label, metrics in (
        ("ordinary", ordinary_metrics),
        ("weighted", weighted_metrics),
    ):
        print(
            f"{label:10s}"
            f"   {metrics['average_size']:.6f}"
            f"       {metrics['singleton_rate']:.6%}"
            f"       {metrics['singleton_error']:.6%}"
        )

    ambiguous_error = 0b0101010
    ambiguous_syndrome = syndrome(ambiguous_error, distance)
    record = cache[ambiguous_syndrome]
    ordinary_set = prediction_set(record, ordinary_threshold)
    weighted_set = prediction_set(record, weighted_threshold)
    weighted_corrections = tuple(
        record["candidates"][label] for label in weighted_set
    )
    fidelities = tuple(
        qiskit_recovery_fidelity(
            distance,
            ambiguous_error,
            correction,
        )
        for correction in weighted_corrections
    )

    print("\nDrift-sensitive ambiguous syndrome")
    print(f"true error:                   {ambiguous_error:07b}")
    print(f"syndrome:                     {''.join(map(str, ambiguous_syndrome))}")
    print(
        "source-model posterior:     "
        f"[{record['posterior'][0]:.9f}, {record['posterior'][1]:.9f}]"
    )
    print(f"ordinary conformal set:       {ordinary_set}")
    print(f"importance-weighted set:      {weighted_set}")
    print(
        "weighted-set fidelities:    "
        + ", ".join(f"{value:.9f}" for value in fidelities)
    )

    print("\nInterpretation")
    print(
        "Ordinary calibration undercovers after the physical-noise "
        "distribution changes."
    )
    print(
        "Likelihood-ratio weighting shifts the empirical score CDF toward "
        "the target hardware distribution, at the cost of reduced ESS."
    )


if __name__ == "__main__":
    main()
