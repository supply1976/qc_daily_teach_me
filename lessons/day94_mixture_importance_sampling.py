"""Day 94: Mixture importance sampling for multimodal logical failures."""

import numpy as np

from day91_repetition_code_distance_scaling import (
    exact_logical_error_rate,
    recovered_fidelity,
)


def burst_masks(distance, burst_length):
    """All contiguous bursts of a fixed length on an open 1D chain."""
    base = (1 << burst_length) - 1
    return np.array(
        [base << start for start in range(distance - burst_length + 1)],
        dtype=np.uint64,
    )


def burst_probabilities(masks, allowed_bursts):
    """Probability under the uniform contiguous-burst generator."""
    return np.isin(masks, allowed_bursts).astype(float) / len(
        allowed_bursts
    )


def iid_pattern_probabilities(weights, distance, error_rate):
    """Probability of each individual bit pattern under iid Bernoulli noise."""
    return (
        error_rate**weights
        * (1.0 - error_rate) ** (distance - weights)
    )


def sample_proposal(
    distance,
    proposal_error_rate,
    iid_mixture_weight,
    allowed_bursts,
    trials,
    rng,
):
    """Sample alpha Q_iid + (1-alpha) Q_burst."""
    choose_iid = rng.random(trials) < iid_mixture_weight
    masks = np.empty(trials, dtype=np.uint64)

    iid_count = int(np.sum(choose_iid))
    if iid_count:
        bits = rng.random((iid_count, distance)) < proposal_error_rate
        powers = np.left_shift(
            np.uint64(1),
            np.arange(distance, dtype=np.uint64),
        )
        masks[choose_iid] = bits.astype(np.uint64) @ powers

    burst_count = trials - iid_count
    if burst_count:
        burst_indices = rng.integers(
            0,
            len(allowed_bursts),
            size=burst_count,
        )
        masks[~choose_iid] = allowed_bursts[burst_indices]

    # NumPy 2.x provides a vectorized bit-count ufunc only on some builds;
    # unpacking 21 low bits is portable and still small for this lesson.
    bit_positions = np.arange(distance, dtype=np.uint64)
    weights = np.sum(
        (masks[:, None] >> bit_positions) & np.uint64(1),
        axis=1,
    ).astype(int)
    return masks, weights, iid_count, burst_count


def mixture_importance_sampling(
    distance,
    target_error_rate,
    target_burst_rate,
    proposal_error_rate,
    iid_mixture_weight,
    burst_length,
    trials,
    rng,
):
    """Estimate target logical failure using an iid/burst proposal mixture."""
    allowed_bursts = burst_masks(distance, burst_length)
    masks, weights, iid_count, burst_count = sample_proposal(
        distance,
        proposal_error_rate,
        iid_mixture_weight,
        allowed_bursts,
        trials,
        rng,
    )

    burst_probability = burst_probabilities(masks, allowed_bursts)
    target_probability = (
        (1.0 - target_burst_rate)
        * iid_pattern_probabilities(weights, distance, target_error_rate)
        + target_burst_rate * burst_probability
    )
    proposal_probability = (
        iid_mixture_weight
        * iid_pattern_probabilities(
            weights,
            distance,
            proposal_error_rate,
        )
        + (1.0 - iid_mixture_weight) * burst_probability
    )

    failures = weights > distance // 2
    contributions = failures * target_probability / proposal_probability
    estimate = float(np.mean(contributions))
    standard_error = float(
        np.std(contributions, ddof=1) / np.sqrt(trials)
    )
    nonzero = contributions[failures]
    ess = float(np.sum(nonzero) ** 2 / np.sum(nonzero**2))

    sampled_burst_patterns = int(
        np.sum(failures & (burst_probability > 0.0))
    )
    sampled_other_failures = int(
        np.sum(failures & (burst_probability == 0.0))
    )
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ess": ess,
        "iid_draws": iid_count,
        "burst_draws": burst_count,
        "burst_failures": sampled_burst_patterns,
        "other_failures": sampled_other_failures,
    }


def main():
    distance = 21
    target_error_rate = 0.02
    burst_length = distance // 2 + 1
    target_burst_rate = 6.0e-14
    proposal_error_rate = 0.5246
    trials = 300_000
    rng = np.random.default_rng(9401)

    iid_logical_error = exact_logical_error_rate(
        distance,
        target_error_rate,
    )
    # Every allowed burst has weight 11 and therefore causes logical failure.
    exact_mixture_error = (
        (1.0 - target_burst_rate) * iid_logical_error
        + target_burst_rate
    )

    print("Mixture importance sampling for multimodal failures")
    print(f"distance d:                    {distance}")
    print(f"target iid error p:            {target_error_rate:.3f}")
    print(f"target burst probability r:    {target_burst_rate:.3e}")
    print(f"burst length:                  {burst_length}")
    print(f"iid logical contribution:      {iid_logical_error:.12e}")
    print(f"burst logical contribution:    {target_burst_rate:.12e}")
    print(f"exact total logical error:      {exact_mixture_error:.12e}")

    print("\nProposal mixture comparison")
    print("alpha   estimate          rel. SE     ESS      burst-mode   other")
    print("-" * 76)
    for iid_mixture_weight in (1.0, 0.50, 0.05):
        result = mixture_importance_sampling(
            distance=distance,
            target_error_rate=target_error_rate,
            target_burst_rate=target_burst_rate,
            proposal_error_rate=proposal_error_rate,
            iid_mixture_weight=iid_mixture_weight,
            burst_length=burst_length,
            trials=trials,
            rng=rng,
        )
        relative_se = result["standard_error"] / result["estimate"]
        print(
            f"{iid_mixture_weight:5.2f}"
            f"   {result['estimate']:.12e}"
            f"   {relative_se:8.3%}"
            f"   {result['ess']:8.1f}"
            f"   {result['burst_failures']:10d}"
            f"   {result['other_failures']:7d}"
        )

    print("\nQiskit contiguous-burst check")
    check_distance = 7
    for length in (3, 4):
        mask = (1 << length) - 1
        _, fidelity = recovered_fidelity(check_distance, mask)
        print(
            f"d={check_distance}, burst length={length}: "
            f"logical fidelity={fidelity:.9f}"
        )

    print("\nInterpretation")
    print(
        "alpha=1 uses only the iid proposal and rarely hits a specific "
        "contiguous-burst mode."
    )
    print(
        "A defensive mixture keeps nonzero iid support while explicitly "
        "covering the burst mode, reducing mode-dropping variance."
    )


if __name__ == "__main__":
    main()
