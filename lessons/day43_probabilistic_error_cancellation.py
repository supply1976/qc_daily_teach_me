"""Day 43: probabilistic error cancellation with quasiprobabilities."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp


Z = SparsePauliOp.from_list([("Z", 1.0)])


def prepared_state(theta):
    """Prepare |psi> = Ry(theta)|0>."""
    circuit = QuantumCircuit(1)
    circuit.ry(float(theta), 0)
    return DensityMatrix.from_instruction(circuit)


def apply_x(density):
    circuit = QuantumCircuit(1)
    circuit.x(0)
    return density.evolve(circuit)


def bit_flip_channel(density, error_probability):
    """N_p(rho) = (1-p) rho + p X rho X."""
    p = float(error_probability)
    flipped = apply_x(density)
    return DensityMatrix(
        (1.0 - p) * density.data + p * flipped.data
    )


def expectation_z(density):
    return float(np.real(density.expectation_value(Z)))


def pec_coefficients(error_probability):
    """Return coefficients of N_p^{-1} = eta_I I + eta_X X."""
    p = float(error_probability)
    if not 0.0 <= p < 0.5:
        raise ValueError("PEC requires 0 <= error_probability < 0.5")

    denominator = 1.0 - 2.0 * p
    return np.array([
        (1.0 - p) / denominator,
        -p / denominator,
    ])


def exact_pec_state(noisy_density, coefficients):
    """Apply the inverse channel as a signed linear combination."""
    identity_branch = noisy_density.data
    x_branch = apply_x(noisy_density).data
    return DensityMatrix(
        coefficients[0] * identity_branch
        + coefficients[1] * x_branch
    )


def sample_z(density, shots, rng):
    probability_zero = float(np.real(density.data[0, 0]))
    return np.where(
        rng.random(shots) < probability_zero,
        1.0,
        -1.0,
    )


def sampled_raw_expectation(noisy_density, shots, rng):
    return float(np.mean(sample_z(noisy_density, shots, rng)))


def sampled_pec_expectation(
    noisy_density,
    coefficients,
    shots,
    rng,
):
    """Sample physical branches and attach their signed PEC weights."""
    gamma = float(np.sum(np.abs(coefficients)))
    branch_probabilities = np.abs(coefficients) / gamma
    branch_signs = np.sign(coefficients)

    branches = rng.choice(
        2,
        size=shots,
        p=branch_probabilities,
    )
    contributions = np.empty(shots)

    physical_states = [
        noisy_density,
        apply_x(noisy_density),
    ]

    for branch in range(2):
        selected = branches == branch
        count = int(np.sum(selected))
        if count == 0:
            continue

        outcomes = sample_z(
            physical_states[branch],
            count,
            rng,
        )
        contributions[selected] = (
            gamma * branch_signs[branch] * outcomes
        )

    return float(np.mean(contributions))


def summarize(values, ideal):
    values = np.asarray(values)
    mean = float(np.mean(values))
    bias = mean - ideal
    standard_deviation = float(np.std(values, ddof=1))
    rmse = float(np.sqrt(np.mean((values - ideal) ** 2)))
    return mean, bias, standard_deviation, rmse


def main():
    theta = 0.9
    error_probability = 0.12
    shots = 4096
    repetitions = 500

    ideal_state = prepared_state(theta)
    noisy_state = bit_flip_channel(
        ideal_state,
        error_probability,
    )

    coefficients = pec_coefficients(error_probability)
    gamma = float(np.sum(np.abs(coefficients)))
    mitigated_state = exact_pec_state(
        noisy_state,
        coefficients,
    )

    ideal = expectation_z(ideal_state)
    noisy = expectation_z(noisy_state)
    exact_pec = expectation_z(mitigated_state)

    print(f"State: Ry({theta})|0>")
    print(f"Bit-flip probability p: {error_probability:.3f}")
    print(f"Ideal <Z>:      {ideal:+.9f}")
    print(f"Noisy <Z>:      {noisy:+.9f}")
    print(f"Analytic noisy: {(1-2*error_probability)*ideal:+.9f}")
    print(f"Exact PEC <Z>:  {exact_pec:+.9f}")

    print("\nInverse-channel quasiprobabilities:")
    print(f"  eta_I = {coefficients[0]:+.9f}")
    print(f"  eta_X = {coefficients[1]:+.9f}")
    print(f"  sum eta = {np.sum(coefficients):.9f}")
    print(f"  gamma = sum |eta| = {gamma:.9f}")

    rng = np.random.default_rng(20260929)
    raw_results = []
    pec_results = []

    for _ in range(repetitions):
        raw_results.append(
            sampled_raw_expectation(
                noisy_state,
                shots,
                rng,
            )
        )
        pec_results.append(
            sampled_pec_expectation(
                noisy_state,
                coefficients,
                shots,
                rng,
            )
        )

    raw_summary = summarize(raw_results, ideal)
    pec_summary = summarize(pec_results, ideal)

    raw_predicted_std = np.sqrt(
        (1.0 - noisy**2) / shots
    )
    pec_predicted_std = np.sqrt(
        (gamma**2 - ideal**2) / shots
    )

    print(
        f"\nFinite-shot results: {shots} shots, "
        f"{repetitions} repetitions"
    )
    print(" method       mean          bias          std         RMSE")
    print("-" * 67)
    for name, result in [
        ("raw", raw_summary),
        ("PEC", pec_summary),
    ]:
        print(
            f" {name:6s}  {result[0]:+.9f}  {result[1]:+.3e}"
            f"  {result[2]:.6f}  {result[3]:.6f}"
        )

    print("\nPredicted standard deviations:")
    print(f"  raw: {raw_predicted_std:.6f}")
    print(f"  PEC: {pec_predicted_std:.6f}")

    print("\nDepth-dependent PEC cost if every layer has this noise:")
    print(" layers    total gamma    approximate shot multiplier gamma^2")
    print("-" * 64)
    for layers in [1, 2, 5, 10, 20]:
        total_gamma = gamma**layers
        print(
            f" {layers:6d}    {total_gamma:11.6f}"
            f"              {total_gamma**2:12.3f}"
        )


if __name__ == "__main__":
    main()
