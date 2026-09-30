"""Day 44: symmetry verification by Bell-parity postselection."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    SparsePauliOp,
    state_fidelity,
)


XX = SparsePauliOp.from_list([("XX", 1.0)])
ZZ = SparsePauliOp.from_list([("ZZ", 1.0)])


def bell_state():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return DensityMatrix.from_instruction(circuit)


def apply_pauli(density, qubit, label):
    circuit = QuantumCircuit(2)

    if label == "X":
        circuit.x(qubit)
    elif label == "Y":
        circuit.y(qubit)
    elif label == "Z":
        circuit.z(qubit)
    else:
        raise ValueError("label must be X, Y, or Z")

    return density.evolve(circuit)


def local_depolarizing_channel(density, error_probability):
    """Apply an independent I/X/Y/Z Pauli channel to each qubit."""
    p = float(error_probability)
    current = density

    for qubit in range(2):
        updated = (1.0 - p) * current.data

        for label in ["X", "Y", "Z"]:
            updated += (
                p / 3.0
                * apply_pauli(current, qubit, label).data
            )

        current = DensityMatrix(updated)

    return current


def expectation(density, observable):
    return float(np.real(density.expectation_value(observable)))


def even_parity_projector():
    identity = np.eye(4, dtype=complex)
    return 0.5 * (identity + ZZ.to_matrix())


def symmetry_verified_state(density):
    """Project onto the known ZZ=+1 symmetry sector."""
    projector = even_parity_projector()
    projected = projector @ density.data @ projector
    acceptance_probability = float(np.real(np.trace(projected)))

    return (
        DensityMatrix(projected / acceptance_probability),
        acceptance_probability,
    )


def bell_basis_probabilities(density):
    """Measure commuting stabilizers XX and ZZ in one Bell-basis circuit."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    circuit.h(0)
    rotated = density.evolve(circuit)

    probabilities = np.real(np.diag(rotated.data))
    probabilities = np.maximum(probabilities, 0.0)
    return probabilities / np.sum(probabilities)


def sample_stabilizers(density, shots, rng):
    """q0 records XX; q1 records ZZ after inverse Bell preparation."""
    outcomes = rng.choice(
        4,
        size=shots,
        p=bell_basis_probabilities(density),
    )

    q0 = outcomes & 1
    q1 = (outcomes >> 1) & 1

    xx_samples = 1 - 2 * q0
    zz_samples = 1 - 2 * q1
    return xx_samples, zz_samples


def sampled_estimates(density, shots, rng):
    xx_samples, zz_samples = sample_stabilizers(
        density,
        shots,
        rng,
    )

    accepted = zz_samples == 1
    raw_xx = float(np.mean(xx_samples))
    verified_xx = float(np.mean(xx_samples[accepted]))
    acceptance = float(np.mean(accepted))
    return raw_xx, verified_xx, acceptance


def summarize(values, ideal):
    values = np.asarray(values)
    mean = float(np.mean(values))
    bias = mean - ideal
    standard_deviation = float(np.std(values, ddof=1))
    rmse = float(np.sqrt(np.mean((values - ideal) ** 2)))
    return mean, bias, standard_deviation, rmse


def main():
    error_probability = 0.12
    shots = 4096
    repetitions = 500

    ideal_state = bell_state()
    noisy_state = local_depolarizing_channel(
        ideal_state,
        error_probability,
    )
    verified_state, exact_acceptance = symmetry_verified_state(
        noisy_state
    )

    ideal_xx = expectation(ideal_state, XX)
    raw_xx = expectation(noisy_state, XX)
    verified_xx = expectation(verified_state, XX)

    print("Target state: |Phi+> = (|00> + |11>) / sqrt(2)")
    print(f"Local depolarizing probability: {error_probability:.3f}")
    print(f"Ideal <XX>:              {ideal_xx:+.9f}")
    print(f"Raw noisy <XX>:          {raw_xx:+.9f}")
    print(f"Verified <XX>:           {verified_xx:+.9f}")
    print(f"Exact acceptance rate:   {exact_acceptance:.9f}")
    print(
        f"Raw state fidelity:      "
        f"{state_fidelity(ideal_state, noisy_state):.9f}"
    )
    print(
        f"Verified state fidelity: "
        f"{state_fidelity(ideal_state, verified_state):.9f}"
    )

    shrinkage = 1.0 - 4.0 * error_probability / 3.0
    analytic_raw = shrinkage**2
    analytic_acceptance = 0.5 * (1.0 + analytic_raw)
    analytic_verified = analytic_raw / analytic_acceptance

    print("\nAnalytic checks:")
    print(f"  Pauli shrinkage lambda: {shrinkage:.9f}")
    print(f"  raw <XX> = lambda^2:    {analytic_raw:.9f}")
    print(f"  acceptance:             {analytic_acceptance:.9f}")
    print(f"  verified <XX>:          {analytic_verified:.9f}")

    rng = np.random.default_rng(20260930)
    raw_results = []
    verified_results = []
    acceptance_results = []

    for _ in range(repetitions):
        raw, verified, acceptance = sampled_estimates(
            noisy_state,
            shots,
            rng,
        )
        raw_results.append(raw)
        verified_results.append(verified)
        acceptance_results.append(acceptance)

    raw_summary = summarize(raw_results, ideal_xx)
    verified_summary = summarize(
        verified_results,
        ideal_xx,
    )

    raw_predicted_std = np.sqrt(
        (1.0 - raw_xx**2) / shots
    )
    verified_predicted_std = np.sqrt(
        (1.0 - verified_xx**2)
        / (shots * exact_acceptance)
    )

    print(
        f"\nFinite-shot results: {shots} shots, "
        f"{repetitions} repetitions"
    )
    print(" method       mean          bias          std         RMSE")
    print("-" * 67)
    for name, result in [
        ("raw", raw_summary),
        ("verified", verified_summary),
    ]:
        print(
            f" {name:8s} {result[0]:+.9f}  {result[1]:+.3e}"
            f"  {result[2]:.6f}  {result[3]:.6f}"
        )

    print("\nAcceptance statistics:")
    print(
        f"  mean acceptance rate: "
        f"{np.mean(acceptance_results):.6f}"
    )
    print(
        f"  mean accepted shots:  "
        f"{shots*np.mean(acceptance_results):.1f} / {shots}"
    )
    print(
        f"  shots needed per accepted shot: "
        f"{1.0/exact_acceptance:.6f}"
    )
    print("\nPredicted standard deviations:")
    print(f"  raw:      {raw_predicted_std:.6f}")
    print(f"  verified: {verified_predicted_std:.6f}")

    print("\nErrors invisible to the ZZ symmetry check:")
    phase_error = apply_pauli(ideal_state, 0, "Z")
    phase_verified, phase_acceptance = symmetry_verified_state(
        phase_error
    )
    print(f"  Z error acceptance: {phase_acceptance:.1f}")
    print(
        f"  <XX> after accepted Z error: "
        f"{expectation(phase_verified, XX):+.1f}"
    )


if __name__ == "__main__":
    main()
