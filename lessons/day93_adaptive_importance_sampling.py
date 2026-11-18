"""Day 93: Learn an importance-sampling proposal by weighted cross entropy."""

from math import comb

import numpy as np

from day91_repetition_code_distance_scaling import (
    exact_logical_error_rate,
    recovered_fidelity,
)
from day92_rare_event_importance_sampling import importance_sampling


def weighted_failure_statistics(
    distance,
    physical_error_rate,
    proposal_error_rate,
    trials,
    rng,
):
    """Estimate P(failure) and E_p[weight / d | failure] using q samples."""
    error_weights = rng.binomial(
        distance,
        proposal_error_rate,
        size=trials,
    )
    failures = error_weights > distance // 2

    log_ratio = (
        error_weights
        * np.log(physical_error_rate / proposal_error_rate)
        + (distance - error_weights)
        * np.log(
            (1.0 - physical_error_rate)
            / (1.0 - proposal_error_rate)
        )
    )
    contributions = failures * np.exp(log_ratio)
    normalization = np.sum(contributions)

    if normalization == 0.0:
        return 0.0, np.nan, int(np.sum(failures))

    logical_error = float(np.mean(contributions))
    conditional_error_fraction = float(
        np.sum(contributions * error_weights / distance)
        / normalization
    )
    return (
        logical_error,
        conditional_error_fraction,
        int(np.sum(failures)),
    )


def learn_proposal(
    distance,
    physical_error_rate,
    initial_proposal,
    trials_per_round,
    rounds,
    learning_rate,
    rng,
):
    """Fit q to the target distribution conditioned on logical failure."""
    proposal = float(initial_proposal)
    history = []

    for iteration in range(rounds):
        estimate, fitted, failures = weighted_failure_statistics(
            distance,
            physical_error_rate,
            proposal,
            trials_per_round,
            rng,
        )
        if not np.isfinite(fitted):
            raise RuntimeError(
                "No failure samples: choose a larger initial proposal."
            )

        updated = (
            (1.0 - learning_rate) * proposal
            + learning_rate * fitted
        )
        history.append((
            iteration,
            proposal,
            failures,
            estimate,
            fitted,
            updated,
        ))
        proposal = updated

    return proposal, history


def exact_conditional_error_fraction(distance, physical_error_rate):
    """Compute E_p[weight / d | logical failure] by exact summation."""
    probabilities = []
    weighted_probabilities = []
    for weight in range(distance // 2 + 1, distance + 1):
        probability = (
            comb(distance, weight)
            * physical_error_rate**weight
            * (1.0 - physical_error_rate) ** (distance - weight)
        )
        probabilities.append(probability)
        weighted_probabilities.append(weight * probability / distance)
    return float(sum(weighted_probabilities) / sum(probabilities))


def main():
    distance = 21
    physical_error_rate = 0.02
    pilot_trials = 100_000
    evaluation_trials = 300_000
    rng = np.random.default_rng(9301)

    exact_error = exact_logical_error_rate(
        distance,
        physical_error_rate,
    )
    exact_proposal = exact_conditional_error_fraction(
        distance,
        physical_error_rate,
    )

    learned_proposal, history = learn_proposal(
        distance=distance,
        physical_error_rate=physical_error_rate,
        initial_proposal=0.25,
        trials_per_round=pilot_trials,
        rounds=7,
        learning_rate=0.5,
        rng=rng,
    )

    print("Adaptive importance sampling by weighted cross entropy")
    print(f"distance d:                  {distance}")
    print(f"target physical error p:     {physical_error_rate:.3f}")
    print(f"exact logical error P_L:     {exact_error:.12e}")
    print(f"exact conditional-fit q*:    {exact_proposal:.9f}")

    print("\nProposal-learning trajectory")
    print("round   old q   failures   P_L estimate       fitted q    new q")
    print("-" * 75)
    for row in history:
        iteration, old, failures, estimate, fitted, updated = row
        print(
            f"{iteration:5d}   {old:.5f}"
            f"   {failures:8d}"
            f"   {estimate:.6e}"
            f"   {fitted:.6f}   {updated:.6f}"
        )

    print(f"\nlearned proposal q:          {learned_proposal:.9f}")
    print(
        "distance from exact q*:     "
        f"{abs(learned_proposal - exact_proposal):.3e}"
    )

    print("\nIndependent evaluation samples")
    print("proposal       estimate          rel. SE      ESS      failures")
    print("-" * 72)
    for label, proposal in (
        ("low q", 0.25),
        ("learned", learned_proposal),
        ("high q", 0.80),
    ):
        estimate, standard_error, failures, ess = importance_sampling(
            distance,
            physical_error_rate,
            proposal,
            evaluation_trials,
            rng,
        )
        relative_standard_error = standard_error / estimate
        print(
            f"{label:8s}"
            f"   {estimate:.12e}"
            f"   {relative_standard_error:8.3%}"
            f"   {ess:8.1f}   {failures:8d}"
        )

    print("\nQiskit decoder boundary check")
    for error_weight in (3, 4):
        error_mask = (1 << error_weight) - 1
        _, fidelity = recovered_fidelity(7, error_mask)
        print(
            f"d=7, error weight={error_weight}: "
            f"logical fidelity={fidelity:.9f}"
        )

    print("\nInterpretation")
    print(
        "q* is the weighted maximum-likelihood Bernoulli model of "
        "the target noise conditioned on decoder failure."
    )
    print(
        "Pilot samples train the proposal; fresh independent samples "
        "produce the final error-rate estimate."
    )


if __name__ == "__main__":
    main()
