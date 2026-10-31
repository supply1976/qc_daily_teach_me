"""Day 75: Hashing yield and useful entanglement rate of a repeater."""

import numpy as np
from qiskit.quantum_info import entropy

from day74_repeater_memory_cutoff import (
    density_from_bell_probabilities,
    simulate_cutoff_policy,
)


def werner_probabilities(fidelity):
    error_probability = (1.0 - fidelity) / 3.0
    return np.array([
        fidelity,
        error_probability,
        error_probability,
        error_probability,
    ])


def shannon_entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    nonzero = probabilities > 0.0
    return float(-np.sum(
        probabilities[nonzero]
        * np.log2(probabilities[nonzero])
    ))


def hashing_yield(fidelity):
    """One-way hashing lower bound in ebits per Bell-diagonal pair."""
    bell_entropy = shannon_entropy(
        werner_probabilities(fidelity)
    )
    return max(0.0, 1.0 - bell_entropy)


def exact_cutoff_statistics(
    link_probability,
    initial_fidelity,
    memory_lifetime,
    cutoff,
):
    """Exact renewal statistics for a two-link repeater cutoff policy."""
    p = link_probability
    q = 1.0 - p
    visibility = (4.0 * initial_fidelity - 1.0) / 3.0
    decay_per_cycle = np.exp(-1.0 / memory_lifetime)

    # Conditioned on a cycle in which at least one elementary link succeeds.
    simultaneous = p / (2.0 - p)
    exactly_one = 2.0 * q / (2.0 - p)
    time_to_first_link = 1.0 / (p * (2.0 - p))

    if cutoff is None:
        success_per_episode = 1.0
        episode_cycles = time_to_first_link + exactly_one / p
        mean_wait = 2.0 * q / (p * (2.0 - p))

        # Distribution: P(D=0)=p/(2-p),
        # P(D=d)=2p*q**d/(2-p), d>=1.
        mean_decay = (
            p / (2.0 - p)
            * (1.0 + q * decay_per_cycle)
            / (1.0 - q * decay_per_cycle)
        )
    else:
        ages = np.arange(1, cutoff + 1, dtype=float)
        age_weights = (
            exactly_one
            * p
            * q ** (ages - 1.0)
        )
        success_per_episode = (
            simultaneous + np.sum(age_weights)
        )
        episode_cycles = (
            time_to_first_link
            + exactly_one * (1.0 - q**cutoff) / p
        )
        mean_wait = float(
            np.sum(ages * age_weights)
            / success_per_episode
        )
        mean_decay = float(
            (
                simultaneous
                + np.sum(
                    age_weights
                    * decay_per_cycle**ages
                )
            )
            / success_per_episode
        )

    pair_rate = success_per_episode / episode_cycles
    mean_fidelity = (
        0.25
        + 0.75 * visibility**2 * mean_decay
    )
    yield_per_pair = hashing_yield(mean_fidelity)
    ebit_rate = pair_rate * yield_per_pair

    return {
        "cutoff": cutoff,
        "mean_wait": mean_wait,
        "fidelity": mean_fidelity,
        "pair_rate": pair_rate,
        "hashing_yield": yield_per_pair,
        "ebit_rate": ebit_rate,
    }


def qiskit_coherent_information(fidelity):
    """For Bell-diagonal states S(B)=1, so I(A>B)=1-S(AB)."""
    density = density_from_bell_probabilities(
        werner_probabilities(fidelity)
    )
    joint_entropy = float(entropy(density, base=2))
    return 1.0 - joint_entropy, joint_entropy


def main():
    link_probability = 0.10
    initial_fidelity = 0.99
    memory_lifetime = 15.0

    print("Distillable-entanglement rate for repeater cutoff policies")
    print(f"Link success probability: {link_probability:.3f}")
    print(f"Fresh-link fidelity:      {initial_fidelity:.6f}")
    print(f"Memory lifetime:          {memory_lifetime:.1f} cycles")

    cutoffs = list(range(1, 21)) + [None]
    results = [
        exact_cutoff_statistics(
            link_probability,
            initial_fidelity,
            memory_lifetime,
            cutoff,
        )
        for cutoff in cutoffs
    ]

    print("\nExact rate-quality trade-off")
    print(
        "cutoff   mean F       pair rate     hash yield    "
        "ebit rate"
    )
    print("-" * 70)
    displayed_cutoffs = {1, 2, 3, 4, 5, 6, 8, 10, 15, 20, None}
    for result in results:
        if result["cutoff"] not in displayed_cutoffs:
            continue
        label = (
            "none"
            if result["cutoff"] is None
            else str(result["cutoff"])
        )
        print(
            f"{label:>6s}   "
            f"{result['fidelity']:.9f}   "
            f"{result['pair_rate']:.9f}   "
            f"{result['hashing_yield']:.9f}   "
            f"{result['ebit_rate']:.9f}"
        )

    finite_results = [
        result
        for result in results
        if result["cutoff"] is not None
    ]
    optimum = max(
        finite_results,
        key=lambda result: result["ebit_rate"],
    )
    print("\nOptimal cutoff by hashing-bound ebit rate")
    print(f"Cutoff:            {optimum['cutoff']} cycles")
    print(f"Mean waiting age:  {optimum['mean_wait']:.9f} cycles")
    print(f"Mean fidelity:     {optimum['fidelity']:.9f}")
    print(f"Raw pair rate:     {optimum['pair_rate']:.9f} / cycle")
    print(f"Hashing yield:     {optimum['hashing_yield']:.9f} ebits/pair")
    print(f"Useful ebit rate:  {optimum['ebit_rate']:.9f} ebits/cycle")

    coherent_information, joint_entropy = qiskit_coherent_information(
        optimum["fidelity"]
    )
    print("\nQiskit entropy check at the optimum")
    print(f"S(AB):                    {joint_entropy:.12f} bit")
    print(f"1 - S(AB):                {coherent_information:.12f}")
    print(
        "Classical hashing formula: "
        f"{optimum['hashing_yield']:.12f}"
    )
    print(
        "Absolute discrepancy:      "
        f"{abs(coherent_information - optimum['hashing_yield']):.3e}"
    )

    sampled = simulate_cutoff_policy(
        link_probability,
        initial_fidelity,
        memory_lifetime,
        optimum["cutoff"],
        output_pairs=100_000,
        seed=75,
    )
    sampled_yield = hashing_yield(sampled["mean_fidelity"])
    sampled_ebit_rate = sampled["rate"] * sampled_yield
    print("\nMonte Carlo check (100,000 delivered pairs)")
    print(f"Sampled mean fidelity: {sampled['mean_fidelity']:.9f}")
    print(f"Sampled pair rate:     {sampled['rate']:.9f} / cycle")
    print(f"Sampled ebit rate:     {sampled_ebit_rate:.9f} ebits/cycle")


if __name__ == "__main__":
    main()
