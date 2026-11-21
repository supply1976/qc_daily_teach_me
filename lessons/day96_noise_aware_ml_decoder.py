"""Day 96: Noise-aware maximum-likelihood decoding for correlated errors."""

import numpy as np
from qiskit.quantum_info import state_fidelity

from day91_repetition_code_distance_scaling import (
    apply_x_mask,
    encoded_state,
    minimum_weight_correction,
    syndrome,
)
from day94_mixture_importance_sampling import (
    burst_masks,
    sample_proposal,
)


def syndrome_candidates(syndrome_bits):
    """The two complementary X-error chains having one syndrome."""
    bits = [0]
    for parity in syndrome_bits:
        bits.append(bits[-1] ^ int(parity))
    first = sum(bit << qubit for qubit, bit in enumerate(bits))
    all_qubits = (1 << len(bits)) - 1
    return first, first ^ all_qubits


def error_probability(
    error_mask,
    distance,
    iid_error_rate,
    burst_rate,
    allowed_bursts,
):
    """Probability of one pattern under (1-r) iid(p) + r burst."""
    weight = error_mask.bit_count()
    iid_probability = (
        iid_error_rate**weight
        * (1.0 - iid_error_rate) ** (distance - weight)
    )
    burst_probability = (
        1.0 / len(allowed_bursts)
        if error_mask in allowed_bursts
        else 0.0
    )
    return (
        (1.0 - burst_rate) * iid_probability
        + burst_rate * burst_probability
    )


def maximum_likelihood_correction(
    syndrome_bits,
    iid_error_rate,
    burst_rate,
    allowed_bursts,
):
    """Choose the syndrome-compatible chain with larger model likelihood."""
    first, second = syndrome_candidates(syndrome_bits)
    distance = len(syndrome_bits) + 1
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
    if first_probability >= second_probability:
        return first, first_probability, second_probability
    return second, second_probability, first_probability


def qiskit_recovery_fidelity(distance, error_mask, correction_mask):
    logical_zero = encoded_state(distance)
    corrupted = apply_x_mask(logical_zero, error_mask, distance)
    recovered = apply_x_mask(corrupted, correction_mask, distance)
    return float(state_fidelity(logical_zero, recovered))


def exact_logical_error_rate(
    distance,
    iid_error_rate,
    burst_rate,
    allowed_bursts,
    decoder,
):
    """Enumerate every X pattern and sum probabilities of decoder failures."""
    all_qubits = (1 << distance) - 1
    logical_error = 0.0
    for error_mask in range(1 << distance):
        probability = error_probability(
            error_mask,
            distance,
            iid_error_rate,
            burst_rate,
            allowed_bursts,
        )
        correction = decoder(syndrome(error_mask, distance))
        if (error_mask ^ correction) == all_qubits:
            logical_error += probability
    return logical_error


def main():
    distance = 7
    iid_error_rate = 0.02
    burst_length = 4
    burst_rate = 0.01
    allowed_bursts_array = burst_masks(distance, burst_length)
    allowed_bursts = frozenset(
        int(mask) for mask in allowed_bursts_array
    )

    def mw_decoder(syndrome_bits):
        return minimum_weight_correction(syndrome_bits)

    def ml_decoder(syndrome_bits):
        correction, _, _ = maximum_likelihood_correction(
            syndrome_bits,
            iid_error_rate,
            burst_rate,
            allowed_bursts,
        )
        return correction

    print("Noise-aware maximum-likelihood decoder")
    print(f"distance d:              {distance}")
    print(f"iid physical error p:    {iid_error_rate:.3f}")
    print(f"burst probability r:     {burst_rate:.3f}")
    print(f"modeled burst length:    {burst_length}")

    examples = (
        ("single X", 0b0000001),
        ("modeled burst", 0b0001111),
        ("scattered weight-4", 0b1010101),
    )
    print("\nRepresentative syndromes")
    print("case                 error    MW corr   ML corr   P(ML)/P(other)")
    print("-" * 72)
    for label, error_mask in examples:
        checks = syndrome(error_mask, distance)
        mw_correction = mw_decoder(checks)
        ml_correction, chosen_probability, other_probability = (
            maximum_likelihood_correction(
                checks,
                iid_error_rate,
                burst_rate,
                allowed_bursts,
            )
        )
        print(
            f"{label:20s} {error_mask:07b}"
            f"   {mw_correction:07b}"
            f"   {ml_correction:07b}"
            f"   {chosen_probability / other_probability:12.3e}"
        )

    burst_error = 0b0001111
    burst_checks = syndrome(burst_error, distance)
    mw_burst_correction = mw_decoder(burst_checks)
    ml_burst_correction = ml_decoder(burst_checks)
    print("\nQiskit recovery for a four-qubit contiguous burst")
    print(
        "minimum-weight fidelity: "
        f"{qiskit_recovery_fidelity(distance, burst_error, mw_burst_correction):.9f}"
    )
    print(
        "maximum-likelihood fidelity: "
        f"{qiskit_recovery_fidelity(distance, burst_error, ml_burst_correction):.9f}"
    )

    exact_mw = exact_logical_error_rate(
        distance,
        iid_error_rate,
        burst_rate,
        allowed_bursts,
        mw_decoder,
    )
    exact_ml = exact_logical_error_rate(
        distance,
        iid_error_rate,
        burst_rate,
        allowed_bursts,
        ml_decoder,
    )
    print("\nExact enumeration over all 2^7 error patterns")
    print(f"minimum-weight logical error:   {exact_mw:.12e}")
    print(f"maximum-likelihood logical error:{exact_ml:.12e}")
    print(f"suppression factor:              {exact_mw / exact_ml:.3f}x")

    trials = 1_000_000
    rng = np.random.default_rng(9601)
    sampled_masks, _, _, _ = sample_proposal(
        distance=distance,
        proposal_error_rate=iid_error_rate,
        iid_mixture_weight=1.0 - burst_rate,
        allowed_bursts=allowed_bursts_array,
        trials=trials,
        rng=rng,
    )
    mw_failures = 0
    ml_failures = 0
    all_qubits = (1 << distance) - 1
    for raw_mask in sampled_masks:
        error_mask = int(raw_mask)
        checks = syndrome(error_mask, distance)
        mw_failures += (
            error_mask ^ mw_decoder(checks)
        ) == all_qubits
        ml_failures += (
            error_mask ^ ml_decoder(checks)
        ) == all_qubits

    print(f"\nMonte Carlo check ({trials:,} trials)")
    print(
        "minimum-weight estimate:    "
        f"{mw_failures / trials:.12e} ({mw_failures} failures)"
    )
    print(
        "maximum-likelihood estimate:"
        f"{ml_failures / trials:.12e} ({ml_failures} failures)"
    )

    complement = burst_error ^ all_qubits
    complement_iid_probability = (
        (1.0 - burst_rate)
        * iid_error_rate ** complement.bit_count()
        * (1.0 - iid_error_rate)
        ** (distance - complement.bit_count())
    )
    approximate_crossover = len(allowed_bursts) * complement_iid_probability
    print("\nPrior-dependent decoding boundary")
    print(
        "approximate burst-rate crossover: "
        f"r ≈ {approximate_crossover:.9e}"
    )
    print(
        "Above this scale, the modeled weight-4 burst is more likely "
        "than its weight-3 complement."
    )


if __name__ == "__main__":
    main()
