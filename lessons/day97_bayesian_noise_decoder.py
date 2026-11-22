"""Day 97: Bayesian posterior-predictive decoding under noise uncertainty."""

import numpy as np

from day91_repetition_code_distance_scaling import (
    minimum_weight_correction,
    syndrome,
)
from day94_mixture_importance_sampling import burst_masks
from day96_noise_aware_ml_decoder import (
    exact_logical_error_rate,
    maximum_likelihood_correction,
    qiskit_recovery_fidelity,
)


def beta_posterior(prior_alpha, prior_beta, bursts, trials):
    """Conjugate posterior for an unknown Bernoulli burst rate."""
    alpha = prior_alpha + bursts
    beta = prior_beta + trials - bursts
    mean = alpha / (alpha + beta)
    variance = (
        alpha * beta
        / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    )
    return alpha, beta, mean, float(np.sqrt(variance))


def decoder_from_burst_rate(
    iid_error_rate,
    burst_rate,
    allowed_bursts,
):
    def decoder(syndrome_bits):
        correction, _, _ = maximum_likelihood_correction(
            syndrome_bits,
            iid_error_rate,
            burst_rate,
            allowed_bursts,
        )
        return correction

    return decoder


def main():
    distance = 7
    iid_error_rate = 0.02
    true_burst_rate = 0.01
    burst_length = 4
    allowed_bursts = frozenset(
        int(mask) for mask in burst_masks(distance, burst_length)
    )

    # Prior mean 1 / (1 + 99) = 0.01, but with large uncertainty.
    prior_alpha = 1.0
    prior_beta = 99.0
    calibration_size = 50
    rng = np.random.default_rng(9702)
    calibration_data = rng.binomial(
        1,
        true_burst_rate,
        size=1000,
    )
    observed_bursts = int(np.sum(calibration_data[:calibration_size]))

    mle_rate = observed_bursts / calibration_size
    _, _, posterior_mean, posterior_std = beta_posterior(
        prior_alpha,
        prior_beta,
        observed_bursts,
        calibration_size,
    )

    print("Bayesian posterior-predictive noise-aware decoding")
    print(f"true burst rate:          {true_burst_rate:.6f}")
    print(f"calibration trials:       {calibration_size}")
    print(f"observed bursts:          {observed_bursts}")
    print(f"plug-in MLE rate:         {mle_rate:.9f}")
    print(f"posterior mean rate:      {posterior_mean:.9f}")
    print(f"posterior standard dev.:  {posterior_std:.9f}")

    print("\nSequential calibration")
    print("trials   bursts    MLE r       posterior mean    posterior std")
    print("-" * 69)
    for trials in (50, 100, 200, 500, 1000):
        bursts = int(np.sum(calibration_data[:trials]))
        _, _, mean, std = beta_posterior(
            prior_alpha,
            prior_beta,
            bursts,
            trials,
        )
        print(
            f"{trials:6d}   {bursts:6d}"
            f"   {bursts / trials:.6f}"
            f"       {mean:.9f}"
            f"       {std:.9f}"
        )

    mw_decoder = minimum_weight_correction
    plugin_decoder = decoder_from_burst_rate(
        iid_error_rate,
        mle_rate,
        allowed_bursts,
    )
    bayesian_decoder = decoder_from_burst_rate(
        iid_error_rate,
        posterior_mean,
        allowed_bursts,
    )
    oracle_decoder = decoder_from_burst_rate(
        iid_error_rate,
        true_burst_rate,
        allowed_bursts,
    )

    print("\nLogical error under the true noise distribution")
    print("decoder          assumed r       exact logical error")
    print("-" * 58)
    results = []
    for label, assumed_rate, decoder in (
        ("minimum weight", np.nan, mw_decoder),
        ("plug-in MLE", mle_rate, plugin_decoder),
        ("Bayesian", posterior_mean, bayesian_decoder),
        ("oracle", true_burst_rate, oracle_decoder),
    ):
        logical_error = exact_logical_error_rate(
            distance,
            iid_error_rate,
            true_burst_rate,
            allowed_bursts,
            decoder,
        )
        results.append((label, logical_error))
        rate_text = "n/a" if np.isnan(assumed_rate) else f"{assumed_rate:.9f}"
        print(
            f"{label:15s}  {rate_text:>11s}"
            f"       {logical_error:.12e}"
        )

    burst_error = 0b0001111
    checks = syndrome(burst_error, distance)
    print("\nQiskit recovery for an unseen calibration burst")
    for label, decoder in (
        ("plug-in MLE", plugin_decoder),
        ("Bayesian", bayesian_decoder),
        ("oracle", oracle_decoder),
    ):
        correction = decoder(checks)
        fidelity = qiskit_recovery_fidelity(
            distance,
            burst_error,
            correction,
        )
        print(
            f"{label:12s}: correction={correction:07b}, "
            f"fidelity={fidelity:.9f}"
        )

    approximate_crossover = 2.922062331e-05
    print("\nDecision boundary")
    print(f"burst-rate crossover:   {approximate_crossover:.9e}")
    print(f"plug-in MLE / crossover:{mle_rate / approximate_crossover:.3f}")
    print(
        "posterior / crossover:  "
        f"{posterior_mean / approximate_crossover:.3f}"
    )
    print(
        "For this linear mixture model, integrating over the Beta "
        "posterior is exactly equivalent to using its posterior mean."
    )


if __name__ == "__main__":
    main()
