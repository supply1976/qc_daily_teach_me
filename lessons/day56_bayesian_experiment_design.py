"""Day 56: Bayesian experimental design for quantum phase estimation."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def experiment_circuit(error_angle, sequence_length, control_phase):
    """Ramsey experiment whose Z outcome probes L*error + control_phase."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    for _ in range(sequence_length):
        circuit.rz(float(error_angle), 0)
    circuit.rz(float(control_phase), 0)
    circuit.h(0)
    return circuit


def qiskit_probability_zero(error_angle, sequence_length, control_phase):
    state = Statevector.from_instruction(
        experiment_circuit(
            error_angle,
            sequence_length,
            control_phase,
        )
    )
    return float(state.probabilities()[0])


def probability_zero(
    error_angle,
    sequence_length,
    control_phase,
    depolarizing_alpha,
):
    contrast = depolarizing_alpha**sequence_length
    return 0.5 * (
        1.0
        + contrast
        * np.cos(
            sequence_length * error_angle
            + control_phase
        )
    )


def entropy_bits(probabilities):
    probabilities = np.clip(probabilities, 1e-300, None)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def expected_information_gain(
    posterior,
    error_grid,
    sequence_length,
    control_phase,
    depolarizing_alpha,
):
    """Expected entropy reduction of one binary measurement."""
    likelihood_zero = probability_zero(
        error_grid,
        sequence_length,
        control_phase,
        depolarizing_alpha,
    )
    probability_zero_average = float(
        np.sum(posterior * likelihood_zero)
    )
    probability_one_average = 1.0 - probability_zero_average

    posterior_zero = posterior * likelihood_zero
    posterior_zero /= np.sum(posterior_zero)
    posterior_one = posterior * (1.0 - likelihood_zero)
    posterior_one /= np.sum(posterior_one)

    expected_entropy = (
        probability_zero_average * entropy_bits(posterior_zero)
        + probability_one_average * entropy_bits(posterior_one)
    )
    return entropy_bits(posterior) - expected_entropy


def posterior_summary(error_grid, posterior):
    mean = float(np.sum(error_grid * posterior))
    standard_deviation = float(
        np.sqrt(
            np.sum((error_grid - mean) ** 2 * posterior)
        )
    )
    return {
        "mean": mean,
        "std": standard_deviation,
        "map": float(error_grid[np.argmax(posterior)]),
    }


def update_with_batch(
    posterior,
    likelihood_zero,
    zero_counts,
    one_counts,
):
    log_updated = (
        np.log(np.clip(posterior, 1e-300, None))
        + zero_counts
        * np.log(np.clip(likelihood_zero, 1e-12, 1.0))
        + one_counts
        * np.log(
            np.clip(1.0 - likelihood_zero, 1e-12, 1.0)
        )
    )
    log_updated -= np.max(log_updated)
    updated = np.exp(log_updated)
    return updated / np.sum(updated)


def run_design(
    true_error,
    error_grid,
    candidate_lengths,
    candidate_phases,
    depolarizing_alpha,
    rounds,
    batch_shots,
    seed,
    alias_safe,
):
    posterior = np.ones_like(error_grid)
    posterior /= np.sum(posterior)
    prior_entropy = entropy_bits(posterior)
    rng = np.random.default_rng(seed)
    history = []

    for round_index in range(1, rounds + 1):
        summary = posterior_summary(error_grid, posterior)
        allowed_lengths = candidate_lengths

        if alias_safe:
            # Keep the phase spread below about one radian, preventing a
            # precise but strongly aliased measurement.
            allowed_lengths = [
                length
                for length in candidate_lengths
                if length * summary["std"] <= 1.0
            ]
            if not allowed_lengths:
                allowed_lengths = [min(candidate_lengths)]

        candidates = []
        for length in allowed_lengths:
            for phase in candidate_phases:
                information_gain = expected_information_gain(
                    posterior,
                    error_grid,
                    length,
                    phase,
                    depolarizing_alpha,
                )
                candidates.append((information_gain, length, phase))

        predicted_gain, length, phase = max(candidates)
        true_probability_zero = float(
            probability_zero(
                true_error,
                length,
                phase,
                depolarizing_alpha,
            )
        )
        zero_counts = int(
            rng.binomial(batch_shots, true_probability_zero)
        )
        one_counts = batch_shots - zero_counts

        likelihood_zero = probability_zero(
            error_grid,
            length,
            phase,
            depolarizing_alpha,
        )
        posterior = update_with_batch(
            posterior,
            likelihood_zero,
            zero_counts,
            one_counts,
        )
        summary = posterior_summary(error_grid, posterior)
        history.append({
            "round": round_index,
            "length": length,
            "phase": phase,
            "zero_counts": zero_counts,
            "predicted_gain": predicted_gain,
            "information_gain": (
                prior_entropy - entropy_bits(posterior)
            ),
            **summary,
        })

    return posterior, history


def main():
    true_error = 0.073
    error_grid = np.linspace(-0.15, 0.15, 6001)
    candidate_lengths = [1, 2, 4, 8, 16, 32, 64]
    candidate_phases = np.linspace(
        0.0,
        2.0 * np.pi,
        16,
        endpoint=False,
    )
    depolarizing_alpha = 0.995
    rounds = 12
    batch_shots = 16
    seed = 20261012

    # Validate the Qiskit circuit probability against the ideal formula.
    qiskit_errors = []
    for length in candidate_lengths:
        for phase in candidate_phases:
            qiskit_probability = qiskit_probability_zero(
                true_error,
                length,
                phase,
            )
            analytic_probability = probability_zero(
                true_error,
                length,
                phase,
                depolarizing_alpha=1.0,
            )
            qiskit_errors.append(
                abs(qiskit_probability - analytic_probability)
            )

    adaptive_posterior, adaptive_history = run_design(
        true_error,
        error_grid,
        candidate_lengths,
        candidate_phases,
        depolarizing_alpha,
        rounds,
        batch_shots,
        seed,
        alias_safe=True,
    )
    fixed_posterior, fixed_history = run_design(
        true_error,
        error_grid,
        [16],
        candidate_phases,
        depolarizing_alpha,
        rounds,
        batch_shots,
        seed,
        alias_safe=False,
    )

    adaptive_final = posterior_summary(
        error_grid,
        adaptive_posterior,
    )
    fixed_final = posterior_summary(error_grid, fixed_posterior)
    adaptive_gate_uses = batch_shots * sum(
        result["length"] for result in adaptive_history
    )
    fixed_gate_uses = batch_shots * sum(
        result["length"] for result in fixed_history
    )

    print("Bayesian experimental design for phase estimation")
    print(f"True angle error:       {true_error:+.9f} rad/gate")
    print(f"Rounds:                 {rounds}")
    print(f"Shots per round:        {batch_shots}")
    print(f"Per-gate Bloch shrink:  {depolarizing_alpha:.6f}")
    print(
        "Maximum Qiskit-versus-analytic probability error: "
        f"{max(qiskit_errors):.3e}"
    )

    print("\nAdaptive choices and posterior:")
    print(
        " round    L   phi/pi   zeros"
        "   expected IG    mean error     posterior std   total IG"
    )
    print("-" * 94)
    for result in adaptive_history:
        print(
            f"{result['round']:6d}  {result['length']:3d}"
            f"   {result['phase'] / np.pi:6.3f}"
            f"     {result['zero_counts']:2d}"
            f"      {result['predicted_gain']:.4f}"
            f"       {result['mean']:+.7f}"
            f"       {result['std']:.3e}"
            f"      {result['information_gain']:.3f}"
        )

    print("\nFinal comparison with the same number of shots:")
    print(
        " strategy             mean error      MAP error"
        "       posterior std    gate uses"
    )
    print("-" * 82)
    print(
        f" adaptive L, phi       {adaptive_final['mean']:+.9f}"
        f"   {adaptive_final['map']:+.9f}"
        f"      {adaptive_final['std']:.3e}"
        f"       {adaptive_gate_uses:5d}"
    )
    print(
        f" fixed L=16, adaptive phi {fixed_final['mean']:+.9f}"
        f"   {fixed_final['map']:+.9f}"
        f"      {fixed_final['std']:.3e}"
        f"       {fixed_gate_uses:5d}"
    )
    print(
        "Shot count is equal, but coherent-depth cost is not; "
        "experimental design must state which resource it optimizes."
    )


if __name__ == "__main__":
    main()
