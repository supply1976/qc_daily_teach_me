"""Day 102: Set-valued quantum decoding with split conformal prediction."""

from itertools import product

import numpy as np

from day91_repetition_code_distance_scaling import syndrome
from day94_mixture_importance_sampling import burst_masks, sample_proposal
from day96_noise_aware_ml_decoder import (
    error_probability,
    qiskit_recovery_fidelity,
    syndrome_candidates,
)


def sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + np.exp(-value))
    exponential = np.exp(value)
    return exponential / (1.0 + exponential)


def build_decoder_cache(
    distance,
    iid_error_rate,
    burst_rate,
    allowed_bursts,
    logit_scale,
):
    """Precompute an intentionally overconfident posterior for each syndrome."""
    cache = {}
    for syndrome_bits in product((0, 1), repeat=distance - 1):
        candidates = syndrome_candidates(syndrome_bits)
        joint = np.array([
            error_probability(
                candidate,
                distance,
                iid_error_rate,
                burst_rate,
                allowed_bursts,
            )
            for candidate in candidates
        ])
        exact_llr = float(np.log(joint[0]) - np.log(joint[1]))
        raw_probability_zero = sigmoid(logit_scale * exact_llr)
        cache[syndrome_bits] = {
            "candidates": candidates,
            "posterior": np.array([
                raw_probability_zero,
                1.0 - raw_probability_zero,
            ]),
        }
    return cache


def true_label(error_mask, record):
    """Identify which syndrome-compatible logical class generated the error."""
    if error_mask == record["candidates"][0]:
        return 0
    if error_mask == record["candidates"][1]:
        return 1
    raise ValueError("Error mask is not compatible with its syndrome record.")


def nonconformity_scores(error_masks, distance, cache):
    """Use 1 - predicted probability of the observed class as the score."""
    scores = np.empty(len(error_masks), dtype=float)
    for index, raw_mask in enumerate(error_masks):
        error_mask = int(raw_mask)
        record = cache[syndrome(error_mask, distance)]
        label = true_label(error_mask, record)
        scores[index] = 1.0 - record["posterior"][label]
    return scores


def conformal_quantile(scores, alpha):
    """Finite-sample split-conformal quantile with the higher convention."""
    sample_count = len(scores)
    rank = int(np.ceil((sample_count + 1) * (1.0 - alpha)))
    rank = min(rank, sample_count)
    return float(np.partition(scores, rank - 1)[rank - 1]), rank


def prediction_set(record, score_threshold):
    """Return logical labels whose nonconformity does not exceed q-hat."""
    return tuple(
        label
        for label, probability in enumerate(record["posterior"])
        if 1.0 - probability <= score_threshold + 1.0e-15
    )


def evaluate_sets(
    error_masks,
    distance,
    cache,
    score_threshold,
):
    covered = 0
    total_size = 0
    singleton_count = 0
    singleton_errors = 0

    for raw_mask in error_masks:
        error_mask = int(raw_mask)
        record = cache[syndrome(error_mask, distance)]
        label = true_label(error_mask, record)
        label_set = prediction_set(record, score_threshold)
        covered += label in label_set
        total_size += len(label_set)
        if len(label_set) == 1:
            singleton_count += 1
            singleton_errors += label_set[0] != label

    trials = len(error_masks)
    return {
        "coverage": covered / trials,
        "average_size": total_size / trials,
        "singleton_rate": singleton_count / trials,
        "singleton_error": (
            singleton_errors / singleton_count
            if singleton_count
            else np.nan
        ),
    }


def main():
    distance = 7
    iid_error_rate = 0.25
    burst_rate = 0.10
    burst_length = 4
    logit_scale = 1.50
    calibration_size = 5_000
    test_size = 200_000
    rng = np.random.default_rng(10201)

    allowed_bursts_array = burst_masks(distance, burst_length)
    allowed_bursts = frozenset(
        int(mask) for mask in allowed_bursts_array
    )
    cache = build_decoder_cache(
        distance,
        iid_error_rate,
        burst_rate,
        allowed_bursts,
        logit_scale,
    )

    calibration_masks, _, _, _ = sample_proposal(
        distance,
        iid_error_rate,
        1.0 - burst_rate,
        allowed_bursts_array,
        calibration_size,
        rng,
    )
    test_masks, _, _, _ = sample_proposal(
        distance,
        iid_error_rate,
        1.0 - burst_rate,
        allowed_bursts_array,
        test_size,
        rng,
    )
    scores = nonconformity_scores(
        calibration_masks,
        distance,
        cache,
    )

    print("Split conformal prediction for soft quantum decoding")
    print(f"distance d:                 {distance}")
    print(f"calibration samples:        {calibration_size:,}")
    print(f"independent test samples:   {test_size:,}")
    print(f"overconfident logit scale:  {logit_scale:.2f}")
    print("\nalpha   target   q-hat       coverage   avg size   singleton   singleton error")
    print("-" * 88)

    results = {}
    for alpha in (0.20, 0.10, 0.05, 0.01):
        threshold, rank = conformal_quantile(scores, alpha)
        metrics = evaluate_sets(
            test_masks,
            distance,
            cache,
            threshold,
        )
        results[alpha] = (threshold, rank, metrics)
        print(
            f"{alpha:5.2f}"
            f"   {1.0 - alpha:6.1%}"
            f"   {threshold:9.6f}"
            f"   {metrics['coverage']:8.3%}"
            f"   {metrics['average_size']:8.5f}"
            f"   {metrics['singleton_rate']:8.3%}"
            f"   {metrics['singleton_error']:12.6%}"
        )

    alpha = 0.05
    threshold = results[alpha][0]
    ambiguous_error = 0b0101010
    ambiguous_syndrome = syndrome(ambiguous_error, distance)
    record = cache[ambiguous_syndrome]
    label_set = prediction_set(record, threshold)

    print("\nAmbiguous syndrome at alpha=0.05")
    print(f"true error:                 {ambiguous_error:07b}")
    print(f"syndrome:                   {''.join(map(str, ambiguous_syndrome))}")
    print(
        "candidate errors:           "
        f"{record['candidates'][0]:07b}, {record['candidates'][1]:07b}"
    )
    print(
        "overconfident posterior:    "
        f"[{record['posterior'][0]:.9f}, {record['posterior'][1]:.9f}]"
    )
    print(f"conformal logical set:       {label_set}")
    corrections = tuple(record["candidates"][label] for label in label_set)
    print(
        "correction set:             "
        + ", ".join(f"{correction:07b}" for correction in corrections)
    )
    fidelities = tuple(
        qiskit_recovery_fidelity(
            distance,
            ambiguous_error,
            correction,
        )
        for correction in corrections
    )
    print(
        "Qiskit logical fidelities:  "
        + ", ".join(f"{fidelity:.9f}" for fidelity in fidelities)
    )

    print("\nInterpretation")
    print(
        "The conformal wrapper can return both logical classes when the "
        "calibration data says a singleton would be too risky."
    )
    print(
        "Its guarantee is marginal over exchangeable future errors; it is "
        "not a separate coverage guarantee for every syndrome."
    )


if __name__ == "__main__":
    main()
