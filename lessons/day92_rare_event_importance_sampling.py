"""Day 92: Importance sampling for rare logical failures."""

import numpy as np

from day91_repetition_code_distance_scaling import (
    exact_logical_error_rate,
    recovered_fidelity,
)


def naive_monte_carlo(distance, physical_error_rate, trials, rng):
    """Sample directly from the target physical-noise distribution."""
    weights = rng.binomial(distance, physical_error_rate, size=trials)
    failures = weights > distance // 2
    estimate = float(np.mean(failures))
    standard_error = float(
        np.sqrt(estimate * (1.0 - estimate) / trials)
    )
    return estimate, standard_error, int(np.sum(failures))


def importance_sampling(
    distance,
    physical_error_rate,
    proposal_error_rate,
    trials,
    rng,
):
    """Estimate a Bernoulli-p binomial tail using Bernoulli-q samples."""
    weights = rng.binomial(distance, proposal_error_rate, size=trials)
    failures = weights > distance // 2

    # P_p(pattern) / P_q(pattern); the binomial coefficient cancels.
    log_likelihood_ratio = (
        weights * np.log(physical_error_rate / proposal_error_rate)
        + (distance - weights)
        * np.log(
            (1.0 - physical_error_rate)
            / (1.0 - proposal_error_rate)
        )
    )
    contributions = failures * np.exp(log_likelihood_ratio)

    estimate = float(np.mean(contributions))
    standard_error = float(
        np.std(contributions, ddof=1) / np.sqrt(trials)
    )

    # Effective sample size of the nonzero weighted failure samples.
    nonzero = contributions[failures]
    if nonzero.size == 0:
        effective_samples = 0.0
    else:
        effective_samples = float(
            np.sum(nonzero) ** 2 / np.sum(nonzero**2)
        )

    return (
        estimate,
        standard_error,
        int(np.sum(failures)),
        effective_samples,
    )


def main():
    distance = 21
    physical_error_rate = 0.02
    trials = 300_000
    rng = np.random.default_rng(9201)

    exact = exact_logical_error_rate(distance, physical_error_rate)

    print("Rare logical failures by importance sampling")
    print(f"repetition-code distance d: {distance}")
    print(f"correctable errors t:       {distance // 2}")
    print(f"target physical error p:    {physical_error_rate:.3f}")
    print(f"trials per experiment:      {trials:,}")
    print(f"exact logical error P_L:    {exact:.12e}")
    print(f"expected naive failures:    {trials * exact:.6e}")

    naive, naive_se, naive_failures = naive_monte_carlo(
        distance,
        physical_error_rate,
        trials,
        rng,
    )
    print("\nNaive Monte Carlo")
    print(f"observed failures: {naive_failures}")
    print(f"estimate:          {naive:.12e}")
    print(f"reported SE:       {naive_se:.12e}")
    print(
        "A zero estimate is not evidence that the true logical "
        "error rate is zero."
    )

    print("\nImportance sampling: choose a noisier proposal q")
    print(" q     proposal failures    estimate          rel. error   ESS")
    print("-" * 72)
    for proposal_error_rate in (0.30, 0.50, 0.70):
        estimate, standard_error, failures, ess = importance_sampling(
            distance,
            physical_error_rate,
            proposal_error_rate,
            trials,
            rng,
        )
        relative_error = abs(estimate - exact) / exact
        print(
            f"{proposal_error_rate:4.2f}"
            f"      {failures:7d}"
            f"        {estimate:.12e}"
            f"   {relative_error:9.3%}"
            f"   {ess:8.1f}"
        )
        print(
            f"      one-sigma interval: "
            f"[{estimate - standard_error:.12e}, "
            f"{estimate + standard_error:.12e}]"
        )

    print("\nQiskit check at the decoding boundary")
    check_distance = 7
    for error_weight in (3, 4):
        error_mask = (1 << error_weight) - 1
        correction, fidelity = recovered_fidelity(
            check_distance,
            error_mask,
        )
        print(
            f"d={check_distance}, weight={error_weight}: "
            f"correction={correction:0{check_distance}b}, "
            f"logical fidelity={fidelity:.9f}"
        )

    boundary_weight = distance // 2 + 1
    print("\nWhy q near 0.5 works")
    print(f"first failing weight: {boundary_weight}/{distance}")
    print(
        "boundary fraction:  "
        f"{boundary_weight / distance:.6f}"
    )
    print(
        "The target tail is dominated by its first failing weight; "
        "a proposal near that fraction generates useful failures often."
    )


if __name__ == "__main__":
    main()
