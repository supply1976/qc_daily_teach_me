"""Day 100: Soft decoding with a logical log-likelihood ratio."""

from itertools import product

import numpy as np

from day91_repetition_code_distance_scaling import syndrome
from day94_mixture_importance_sampling import burst_masks
from day96_noise_aware_ml_decoder import (
    error_probability,
    qiskit_recovery_fidelity,
    syndrome_candidates,
)


def soft_decode(
    syndrome_bits,
    distance,
    iid_error_rate,
    burst_rate,
    allowed_bursts,
):
    """Return both logical-class posteriors instead of only an argmax."""
    first, second = syndrome_candidates(syndrome_bits)
    first_probability = error_probability(
        first,
        distance,
        iid_error_rate,
        burst_rate,
        allowed_bursts,
    )
    second_probability = error_probability(
        second,
        distance,
        iid_error_rate,
        burst_rate,
        allowed_bursts,
    )
    normalization = first_probability + second_probability
    posterior = np.array([
        first_probability / normalization,
        second_probability / normalization,
    ])
    log_likelihood_ratio = float(
        np.log(first_probability) - np.log(second_probability)
    )
    candidates = (first, second)
    best_index = int(np.argmax(posterior))
    return {
        "candidates": candidates,
        "posterior": posterior,
        "llr": log_likelihood_ratio,
        "correction": candidates[best_index],
        "confidence": float(posterior[best_index]),
        "syndrome_probability": normalization,
    }


def binary_entropy(probability):
    probability = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return float(
        -probability * np.log2(probability)
        - (1.0 - probability) * np.log2(1.0 - probability)
    )


def decoder_metrics(
    distance,
    iid_error_rate,
    burst_rate,
    allowed_bursts,
    confidence_threshold,
):
    """Exact hard error, entropy, coverage, and selective error."""
    hard_error = 0.0
    conditional_entropy = 0.0
    accepted_mass = 0.0
    accepted_error = 0.0
    records = []

    for syndrome_bits in product((0, 1), repeat=distance - 1):
        result = soft_decode(
            syndrome_bits,
            distance,
            iid_error_rate,
            burst_rate,
            allowed_bursts,
        )
        posterior = result["posterior"]
        syndrome_mass = result["syndrome_probability"]
        error_mass = syndrome_mass * float(np.min(posterior))
        hard_error += error_mass
        conditional_entropy += syndrome_mass * binary_entropy(posterior[0])

        if result["confidence"] >= confidence_threshold:
            accepted_mass += syndrome_mass
            accepted_error += error_mass

        records.append((
            result["confidence"],
            syndrome_bits,
            result,
        ))

    selective_error = (
        accepted_error / accepted_mass
        if accepted_mass > 0.0
        else np.nan
    )
    return {
        "hard_error": hard_error,
        "conditional_entropy": conditional_entropy,
        "coverage": accepted_mass,
        "selective_error": selective_error,
        "records": sorted(records, key=lambda item: item[0]),
    }


def burst_crossover(
    syndrome_bits,
    distance,
    iid_error_rate,
    allowed_bursts,
    modeled_burst,
):
    """Find burst rate where the two candidate errors are equally likely."""
    candidates = syndrome_candidates(syndrome_bits)
    burst_index = candidates.index(modeled_burst)
    lower = 0.0
    upper = 0.1
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        result = soft_decode(
            syndrome_bits,
            distance,
            iid_error_rate,
            middle,
            allowed_bursts,
        )
        if result["posterior"][burst_index] < 0.5:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def main():
    distance = 7
    iid_error_rate = 0.02
    burst_rate = 1.0e-4
    burst_error = 0b0001111
    allowed_bursts = frozenset(
        int(mask) for mask in burst_masks(distance, 4)
    )
    checks = syndrome(burst_error, distance)

    result = soft_decode(
        checks,
        distance,
        iid_error_rate,
        burst_rate,
        allowed_bursts,
    )
    first, second = result["candidates"]

    print("Soft logical decoding by likelihood ratio")
    print(f"syndrome:            {''.join(map(str, checks))}")
    print(f"candidate 0:         {first:07b}")
    print(f"candidate 1:         {second:07b}")
    print(f"posterior P(L=0|s):  {result['posterior'][0]:.9f}")
    print(f"posterior P(L=1|s):  {result['posterior'][1]:.9f}")
    print(f"logical LLR:         {result['llr']:+.9f}")
    print(f"hard correction:     {result['correction']:07b}")
    print(f"decoder confidence:  {result['confidence']:.9f}")

    crossover = burst_crossover(
        checks,
        distance,
        iid_error_rate,
        allowed_bursts,
        burst_error,
    )
    print(f"\nexact LLR crossover r: {crossover:.12e}")
    print("assumed r       LLR             chosen correction   confidence")
    print("-" * 68)
    for assumed_rate in (0.0, 1.0e-5, crossover, 1.0e-4, 1.0e-2):
        row = soft_decode(
            checks,
            distance,
            iid_error_rate,
            assumed_rate,
            allowed_bursts,
        )
        print(
            f"{assumed_rate:10.3e}"
            f"   {row['llr']:+13.6f}"
            f"       {row['correction']:07b}"
            f"          {row['confidence']:.6f}"
        )

    print("\nExact soft-decoder metrics")
    print("threshold   coverage      conditional error")
    print("-" * 52)
    base_metrics = None
    for threshold in (0.50, 0.80, 0.90, 0.99, 0.999):
        metrics = decoder_metrics(
            distance,
            iid_error_rate,
            burst_rate,
            allowed_bursts,
            threshold,
        )
        if base_metrics is None:
            base_metrics = metrics
        print(
            f"{threshold:9.3f}"
            f"   {metrics['coverage']:10.6%}"
            f"   {metrics['selective_error']:.12e}"
        )

    print(
        "\nhard-decision logical error: "
        f"{base_metrics['hard_error']:.12e}"
    )
    print(
        "logical conditional entropy: "
        f"{base_metrics['conditional_entropy']:.12e} bit"
    )

    print("\nThree least-confident syndromes")
    for confidence, syndrome_bits, record in base_metrics["records"][:3]:
        print(
            f"s={''.join(map(str, syndrome_bits))}, "
            f"candidates={record['candidates'][0]:07b}/"
            f"{record['candidates'][1]:07b}, "
            f"confidence={confidence:.9f}"
        )

    fidelity = qiskit_recovery_fidelity(
        distance,
        burst_error,
        result["correction"],
    )
    print("\nQiskit hard-decision check")
    print(f"true error:          {burst_error:07b}")
    print(f"chosen correction:   {result['correction']:07b}")
    print(f"logical fidelity:    {fidelity:.9f}")

    print("\nInterpretation")
    print(
        "The LLR sign chooses a correction; its magnitude measures how "
        "strongly the generative noise prior favors that logical class."
    )
    print(
        "Selective decoding trades coverage for a lower conditional "
        "logical-error rate instead of hiding uncertainty behind argmax."
    )


if __name__ == "__main__":
    main()
