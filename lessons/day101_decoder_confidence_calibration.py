"""Day 101: Calibrate a soft quantum-error decoder with temperature scaling."""

from itertools import product

import numpy as np

from day91_repetition_code_distance_scaling import syndrome
from day94_mixture_importance_sampling import burst_masks
from day96_noise_aware_ml_decoder import (
    error_probability,
    qiskit_recovery_fidelity,
    syndrome_candidates,
)


def sigmoid(value):
    """Numerically stable scalar sigmoid."""
    if value >= 0.0:
        return 1.0 / (1.0 + np.exp(-value))
    exponential = np.exp(value)
    return exponential / (1.0 + exponential)


def true_syndrome_distribution(
    syndrome_bits,
    distance,
    iid_error_rate,
    burst_rate,
    allowed_bursts,
):
    """Return P(s) and the two exact logical-class posteriors."""
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
    syndrome_probability = float(np.sum(joint))
    posterior = joint / syndrome_probability
    exact_llr = float(np.log(joint[0]) - np.log(joint[1]))
    return candidates, syndrome_probability, posterior, exact_llr


def posterior_from_logit(raw_llr, temperature):
    """Convert a raw logical logit into a temperature-scaled posterior."""
    probability_zero = sigmoid(raw_llr / temperature)
    return np.array([probability_zero, 1.0 - probability_zero])


def calibration_records(
    distance,
    iid_error_rate,
    burst_rate,
    allowed_bursts,
    logit_scale,
    temperature,
):
    """Enumerate all syndromes and retain exact outcome probabilities."""
    records = []
    for syndrome_bits in product((0, 1), repeat=distance - 1):
        candidates, mass, true_posterior, exact_llr = (
            true_syndrome_distribution(
                syndrome_bits,
                distance,
                iid_error_rate,
                burst_rate,
                allowed_bursts,
            )
        )
        # Mimic a learned decoder with correct ranking but over-sharp logits.
        raw_llr = logit_scale * exact_llr
        predicted = posterior_from_logit(raw_llr, temperature)
        chosen_index = int(np.argmax(predicted))
        records.append({
            "syndrome": syndrome_bits,
            "candidates": candidates,
            "mass": mass,
            "true": true_posterior,
            "predicted": predicted,
            "raw_llr": raw_llr,
            "chosen_index": chosen_index,
            "confidence": float(predicted[chosen_index]),
            "accuracy": float(true_posterior[chosen_index]),
        })
    return records


def calibration_metrics(records, bin_count=5):
    """Compute proper scoring rules and expected calibration error."""
    negative_log_likelihood = 0.0
    brier_score = 0.0
    logical_error = 0.0

    for record in records:
        mass = record["mass"]
        true = record["true"]
        predicted = np.clip(record["predicted"], 1.0e-15, 1.0)
        negative_log_likelihood += (
            mass * -float(np.sum(true * np.log(predicted)))
        )
        # Binary Brier score for the event L=0.
        brier_score += mass * (
            true[0] * (1.0 - predicted[0]) ** 2
            + true[1] * predicted[0] ** 2
        )
        logical_error += mass * (1.0 - record["accuracy"])

    bins = []
    expected_calibration_error = 0.0
    edges = np.linspace(0.5, 1.0, bin_count + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = [
            record
            for record in records
            if record["confidence"] >= lower
            and (
                record["confidence"] < upper
                or index == bin_count - 1
            )
        ]
        mass = float(sum(record["mass"] for record in selected))
        if mass == 0.0:
            continue
        mean_confidence = sum(
            record["mass"] * record["confidence"]
            for record in selected
        ) / mass
        empirical_accuracy = sum(
            record["mass"] * record["accuracy"]
            for record in selected
        ) / mass
        expected_calibration_error += (
            mass * abs(mean_confidence - empirical_accuracy)
        )
        bins.append((
            lower,
            upper,
            mass,
            mean_confidence,
            empirical_accuracy,
        ))

    return {
        "nll": negative_log_likelihood,
        "brier": brier_score,
        "ece": expected_calibration_error,
        "logical_error": logical_error,
        "bins": bins,
    }


def fit_temperature(record_builder):
    """Minimize exact NLL by golden-section search over log(T)."""
    lower = np.log(0.1)
    upper = np.log(10.0)
    ratio = (np.sqrt(5.0) - 1.0) / 2.0

    def objective(log_temperature):
        records = record_builder(float(np.exp(log_temperature)))
        return calibration_metrics(records)["nll"]

    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = objective(left)
    right_value = objective(right)
    for _ in range(80):
        if left_value < right_value:
            upper = right
            right = left
            right_value = left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        else:
            lower = left
            left = right
            left_value = right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)
    return float(np.exp(0.5 * (lower + upper)))


def selective_metrics(records, threshold):
    """Coverage and error rate after rejecting low-confidence syndromes."""
    accepted = [
        record
        for record in records
        if record["confidence"] >= threshold
    ]
    coverage = float(sum(record["mass"] for record in accepted))
    error_mass = sum(
        record["mass"] * (1.0 - record["accuracy"])
        for record in accepted
    )
    return coverage, float(error_mass / coverage) if coverage else np.nan


def print_reliability_table(label, bins):
    print(f"\n{label}")
    print("confidence bin   probability mass   mean conf.   true accuracy")
    print("-" * 69)
    for lower, upper, mass, confidence, accuracy in bins:
        print(
            f"[{lower:.1f}, {upper:.1f})"
            f"       {mass:12.6%}"
            f"       {confidence:.6f}"
            f"       {accuracy:.6f}"
        )


def main():
    distance = 7
    iid_error_rate = 0.25
    burst_rate = 0.10
    burst_length = 4
    logit_scale = 1.50
    allowed_bursts = frozenset(
        int(mask) for mask in burst_masks(distance, burst_length)
    )

    def build(temperature):
        return calibration_records(
            distance,
            iid_error_rate,
            burst_rate,
            allowed_bursts,
            logit_scale,
            temperature,
        )

    raw_records = build(temperature=1.0)
    fitted_temperature = fit_temperature(build)
    calibrated_records = build(fitted_temperature)
    raw_metrics = calibration_metrics(raw_records)
    calibrated_metrics = calibration_metrics(calibrated_records)

    print("Confidence calibration for a soft logical decoder")
    print(f"distance d:                   {distance}")
    print(f"stress-test iid error p:      {iid_error_rate:.3f}")
    print(f"burst probability r:          {burst_rate:.3f}")
    print(f"artificial logit scale:       {logit_scale:.3f}")
    print(f"fitted temperature:           {fitted_temperature:.9f}")

    print("\nProper scoring rules")
    print("metric                 raw decoder       calibrated")
    print("-" * 58)
    for key, label in (
        ("nll", "negative log likelihood"),
        ("brier", "Brier score"),
        ("ece", "expected calibration error"),
        ("logical_error", "hard logical error"),
    ):
        print(
            f"{label:24s}"
            f" {raw_metrics[key]:.12e}"
            f"   {calibrated_metrics[key]:.12e}"
        )

    print_reliability_table("Raw reliability bins", raw_metrics["bins"])
    print_reliability_table(
        "Temperature-scaled reliability bins",
        calibrated_metrics["bins"],
    )

    print("\nSelective decoding after calibration")
    print("threshold      coverage       conditional logical error")
    print("-" * 62)
    for threshold in (0.50, 0.80, 0.90, 0.95, 0.99):
        coverage, conditional_error = selective_metrics(
            calibrated_records,
            threshold,
        )
        print(
            f"{threshold:9.2f}"
            f"    {coverage:10.6%}"
            f"       {conditional_error:.12e}"
        )

    burst_error = 0b0001111
    burst_syndrome = syndrome(burst_error, distance)
    example = next(
        record
        for record in calibrated_records
        if record["syndrome"] == burst_syndrome
    )
    raw_example = next(
        record
        for record in raw_records
        if record["syndrome"] == burst_syndrome
    )
    correction = example["candidates"][example["chosen_index"]]
    fidelity = qiskit_recovery_fidelity(
        distance,
        burst_error,
        correction,
    )

    print("\nRepresentative four-qubit burst")
    print(f"syndrome:                      {''.join(map(str, burst_syndrome))}")
    print(f"raw confidence:                {raw_example['confidence']:.9f}")
    print(f"calibrated confidence:         {example['confidence']:.9f}")
    print(f"true conditional accuracy:     {example['accuracy']:.9f}")
    print(f"chosen correction:             {correction:07b}")
    print(f"Qiskit logical fidelity:       {fidelity:.9f}")

    print("\nInterpretation")
    print(
        "A positive temperature rescales LLR magnitude without changing "
        "its sign, so hard decisions and logical error stay unchanged."
    )
    print(
        "NLL, Brier score, and ECE improve because reported probabilities "
        "now match the decoder's true conditional correctness."
    )


if __name__ == "__main__":
    main()
