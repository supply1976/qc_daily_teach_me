"""Day 98: Sequential Bayesian calibration around a decoder boundary."""

import numpy as np
from scipy.special import betainc

from day91_repetition_code_distance_scaling import (
    minimum_weight_correction,
    syndrome,
)
from day94_mixture_importance_sampling import burst_masks
from day96_noise_aware_ml_decoder import (
    exact_logical_error_rate,
    qiskit_recovery_fidelity,
)
from day97_bayesian_noise_decoder import (
    beta_posterior,
    decoder_from_burst_rate,
)


def posterior_probability_above(alpha, beta, threshold):
    """P(r > threshold) for r ~ Beta(alpha, beta)."""
    return float(1.0 - betainc(alpha, beta, threshold))


def confidence_action(probability_above, confidence=0.99):
    """Return a conservative calibration/deployment decision."""
    if probability_above >= confidence:
        return "deploy burst-aware decoder"
    if probability_above <= 1.0 - confidence:
        return "deploy minimum-weight decoder"
    return "continue calibration"


def main():
    distance = 7
    iid_error_rate = 0.02
    true_burst_rate = 1.0e-4
    burst_length = 4
    crossover = 2.922062331e-5
    prior_alpha = 1.0
    prior_beta = 19_999.0
    confidence = 0.99
    checkpoints = (0, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000)

    allowed_bursts = frozenset(
        int(mask) for mask in burst_masks(distance, burst_length)
    )
    rng = np.random.default_rng(9801)
    observations = rng.binomial(
        1,
        true_burst_rate,
        size=max(checkpoints),
    )

    print("Decision-aware sequential noise calibration")
    print(f"true burst rate:       {true_burst_rate:.9e}")
    print(f"decoder crossover:     {crossover:.9e}")
    print(f"confidence target:     {confidence:.1%}")
    print(
        "prior mean:           "
        f"{prior_alpha / (prior_alpha + prior_beta):.9e}"
    )

    print("\nSequential posterior decisions")
    print("trials  bursts  posterior mean   P(r>cross)   action")
    print("-" * 79)
    stopping_point = None
    deployment_mean = None
    for trials in checkpoints:
        bursts = int(np.sum(observations[:trials]))
        alpha, beta, mean, _ = beta_posterior(
            prior_alpha,
            prior_beta,
            bursts,
            trials,
        )
        probability_above = posterior_probability_above(
            alpha,
            beta,
            crossover,
        )
        action = confidence_action(probability_above, confidence)
        print(
            f"{trials:6d}  {bursts:6d}"
            f"   {mean:.9e}"
            f"     {probability_above:8.3%}   {action}"
        )
        if stopping_point is None and action != "continue calibration":
            stopping_point = trials
            deployment_mean = mean

    if stopping_point is None:
        raise RuntimeError("Confidence target was not reached.")

    posterior_decoder = decoder_from_burst_rate(
        iid_error_rate,
        deployment_mean,
        allowed_bursts,
    )
    oracle_decoder = decoder_from_burst_rate(
        iid_error_rate,
        true_burst_rate,
        allowed_bursts,
    )

    print("\nDeployment point")
    print(f"selected calibration trials: {stopping_point}")
    print(f"posterior mean at deployment:{deployment_mean:.9e}")

    print("\nLogical error under the true noise distribution")
    print("decoder             exact logical error")
    print("-" * 47)
    for label, decoder in (
        ("minimum weight", minimum_weight_correction),
        ("posterior predictive", posterior_decoder),
        ("oracle", oracle_decoder),
    ):
        logical_error = exact_logical_error_rate(
            distance,
            iid_error_rate,
            true_burst_rate,
            allowed_bursts,
            decoder,
        )
        print(f"{label:22s}{logical_error:.12e}")

    burst_error = 0b0001111
    checks = syndrome(burst_error, distance)
    posterior_correction = posterior_decoder(checks)
    fidelity = qiskit_recovery_fidelity(
        distance,
        burst_error,
        posterior_correction,
    )
    print("\nQiskit deployment check")
    print(f"burst error:          {burst_error:07b}")
    print(f"posterior correction: {posterior_correction:07b}")
    print(f"logical fidelity:     {fidelity:.9f}")

    print("\nWhy calibration can be expensive")
    print(
        "The true rate is rare and only about 3.4 times the crossover. "
        "Early zero counts can move the posterior in the wrong direction."
    )
    print(
        "A stopping rule based on decoder-relevant probability avoids "
        "confusing parameter estimation with decision confidence."
    )


if __name__ == "__main__":
    main()
