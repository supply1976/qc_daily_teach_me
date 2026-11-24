"""Day 99: Choose a calibration batch by decoder-focused information gain."""

import numpy as np
from scipy.special import betainc
from scipy.stats import betabinom

from day91_repetition_code_distance_scaling import syndrome
from day94_mixture_importance_sampling import burst_masks
from day96_noise_aware_ml_decoder import qiskit_recovery_fidelity
from day97_bayesian_noise_decoder import (
    beta_posterior,
    decoder_from_burst_rate,
)
from day98_decision_aware_calibration import (
    posterior_probability_above,
)


def binary_entropy(probability):
    """Entropy in bits of the decoder-side event r > crossover."""
    probability = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return float(
        -probability * np.log2(probability)
        - (1.0 - probability) * np.log2(1.0 - probability)
    )


def expected_decision_entropy(alpha, beta, crossover, batch_size):
    """Average future decision entropy under Beta-Binomial prediction."""
    future_bursts = np.arange(batch_size + 1)
    predictive = betabinom.pmf(
        future_bursts,
        batch_size,
        alpha,
        beta,
    )
    predictive /= np.sum(predictive)

    future_probability_above = 1.0 - betainc(
        alpha + future_bursts,
        beta + batch_size - future_bursts,
        crossover,
    )
    clipped = np.clip(
        future_probability_above,
        1e-15,
        1.0 - 1e-15,
    )
    future_entropies = (
        -clipped * np.log2(clipped)
        - (1.0 - clipped) * np.log2(1.0 - clipped)
    )
    return float(np.sum(predictive * future_entropies))


def main():
    distance = 7
    iid_error_rate = 0.02
    true_burst_rate = 1.0e-4
    crossover = 2.922062331e-5
    prior_alpha = 1.0
    prior_beta = 19_999.0
    initial_trials = 20_000
    candidate_batches = (1_000, 5_000, 20_000, 80_000)
    fixed_setup_cost = 5_000

    rng = np.random.default_rng(9801)
    observations = rng.binomial(
        1,
        true_burst_rate,
        size=initial_trials + max(candidate_batches),
    )
    initial_bursts = int(np.sum(observations[:initial_trials]))
    alpha, beta, current_mean, _ = beta_posterior(
        prior_alpha,
        prior_beta,
        initial_bursts,
        initial_trials,
    )
    current_probability = posterior_probability_above(
        alpha,
        beta,
        crossover,
    )
    current_entropy = binary_entropy(current_probability)

    print("Decision-focused Bayesian experiment design")
    print(f"initial calibration trials: {initial_trials}")
    print(f"observed bursts:            {initial_bursts}")
    print(f"posterior mean r:           {current_mean:.9e}")
    print(f"P(r > crossover):           {current_probability:.6%}")
    print(f"current decision entropy:   {current_entropy:.9f} bit")
    print(f"fixed setup cost:           {fixed_setup_cost} shot-equivalents")

    print("\nCandidate calibration batches")
    print("batch    expected H     information gain    gain / total cost")
    print("-" * 69)
    designs = []
    for batch_size in candidate_batches:
        expected_entropy = expected_decision_entropy(
            alpha,
            beta,
            crossover,
            batch_size,
        )
        information_gain = current_entropy - expected_entropy
        efficiency = information_gain / (
            batch_size + fixed_setup_cost
        )
        designs.append((
            efficiency,
            batch_size,
            expected_entropy,
            information_gain,
        ))
        print(
            f"{batch_size:6d}"
            f"   {expected_entropy:.9f}"
            f"      {information_gain:.9f}"
            f"       {efficiency:.9e}"
        )

    _, selected_batch, _, _ = max(designs)
    print(f"\nselected batch size: {selected_batch}")

    new_slice = observations[
        initial_trials:initial_trials + selected_batch
    ]
    new_bursts = int(np.sum(new_slice))
    total_trials = initial_trials + selected_batch
    total_bursts = initial_bursts + new_bursts
    new_alpha, new_beta, new_mean, _ = beta_posterior(
        prior_alpha,
        prior_beta,
        total_bursts,
        total_trials,
    )
    new_probability = posterior_probability_above(
        new_alpha,
        new_beta,
        crossover,
    )
    print("\nObserved result of the selected batch")
    print(f"new bursts:               {new_bursts}")
    print(f"total calibration trials: {total_trials}")
    print(f"total bursts:             {total_bursts}")
    print(f"updated posterior mean:   {new_mean:.9e}")
    print(f"updated P(r>crossover):   {new_probability:.6%}")
    print(f"updated entropy:          {binary_entropy(new_probability):.9f} bit")

    allowed_bursts = frozenset(
        int(mask) for mask in burst_masks(distance, 4)
    )
    posterior_decoder = decoder_from_burst_rate(
        iid_error_rate,
        new_mean,
        allowed_bursts,
    )
    burst_error = 0b0001111
    correction = posterior_decoder(syndrome(burst_error, distance))
    fidelity = qiskit_recovery_fidelity(
        distance,
        burst_error,
        correction,
    )
    print("\nQiskit deployment check")
    print(f"posterior correction: {correction:07b}")
    print(f"logical fidelity:     {fidelity:.9f}")

    print("\nInterpretation")
    print(
        "Small batches have high information per raw shot, while large "
        "batches buy more total entropy reduction."
    )
    print(
        "Including a fixed setup overhead makes the intermediate batch "
        "the most efficient decision-focused design."
    )


if __name__ == "__main__":
    main()
