"""Day 45: two-copy virtual distillation and its coherent-error floor."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp


Z = SparsePauliOp.from_list([("Z", 1.0)])


def zero_state():
    return DensityMatrix.from_instruction(QuantumCircuit(1))


def apply_x(density):
    circuit = QuantumCircuit(1)
    circuit.x(0)
    return density.evolve(circuit)


def incoherent_bit_flip_state(error_probability):
    """rho = (1-p)|0><0| + p|1><1|."""
    p = float(error_probability)
    ideal = zero_state()
    flipped = apply_x(ideal)
    return DensityMatrix(
        (1.0 - p) * ideal.data + p * flipped.data
    )


def coherent_rotation_state(angle):
    """A pure but coherently misrotated state Ry(angle)|0>."""
    circuit = QuantumCircuit(1)
    circuit.ry(float(angle), 0)
    return DensityMatrix.from_instruction(circuit)


def expectation_z(density):
    return float(np.real(density.expectation_value(Z)))


def virtual_distilled_state(density, copies):
    powered = np.linalg.matrix_power(density.data, copies)
    return DensityMatrix(powered / np.trace(powered))


def purity(density):
    return float(np.real(np.trace(density.data @ density.data)))


def two_copy_hadamard_test(density, include_z):
    """Return the ancilla distribution for Tr(S rho⊗rho) or Tr(S Zrho⊗rho)."""
    ancilla = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
    ], dtype=complex)

    # Qiskit orders the basis as |q2 q1 q0>; q0 is the ancilla.
    three_qubit_density = DensityMatrix(
        np.kron(density.data, np.kron(density.data, ancilla))
    )

    circuit = QuantumCircuit(3)
    circuit.h(0)

    if include_z:
        # On the ancilla-|1> branch, apply Z to copy A.
        circuit.cz(0, 1)

    circuit.cswap(0, 1, 2)
    circuit.h(0)

    evolved = three_qubit_density.evolve(circuit)
    diagonal = np.real(np.diag(evolved.data))
    indices = np.arange(8)
    probability_zero = float(np.sum(diagonal[(indices & 1) == 0]))

    return np.array([
        probability_zero,
        1.0 - probability_zero,
    ])


def ancilla_expectation(probabilities):
    return float(probabilities[0] - probabilities[1])


def sample_pm1(probabilities, shots, rng):
    zeros = rng.binomial(shots, probabilities[0])
    return float((2 * zeros - shots) / shots)


def sample_raw_z(density, shots, rng):
    probability_zero = float(np.real(density.data[0, 0]))
    zeros = rng.binomial(shots, probability_zero)
    return float((2 * zeros - shots) / shots)


def summarize(values, target):
    values = np.asarray(values)
    mean = float(np.mean(values))
    bias = mean - target
    standard_deviation = float(np.std(values, ddof=1))
    rmse = float(np.sqrt(np.mean((values - target) ** 2)))
    return mean, bias, standard_deviation, rmse


def main():
    error_probability = 0.20
    coherent_angle = 0.40
    shots = 4096
    repetitions = 500

    ideal = zero_state()
    incoherent = incoherent_bit_flip_state(
        error_probability
    )

    print("Incoherent mixture:")
    print(f"  bit-flip probability: {error_probability:.3f}")
    print(f"  ideal <Z>:            {expectation_z(ideal):+.9f}")
    print(f"  noisy <Z>:            {expectation_z(incoherent):+.9f}")
    print(f"  purity Tr(rho^2):     {purity(incoherent):.9f}")

    print("\nVirtual-distillation order:")
    print(" copies       <Z>         dominant eigenvalue")
    print("-" * 51)

    for copies in [1, 2, 3, 4, 8]:
        distilled = virtual_distilled_state(
            incoherent,
            copies,
        )
        eigenvalues = np.linalg.eigvalsh(distilled.data)
        print(
            f" {copies:6d}   {expectation_z(distilled):+.9f}"
            f"       {np.max(eigenvalues):.9f}"
        )

    purity_probabilities = two_copy_hadamard_test(
        incoherent,
        include_z=False,
    )
    numerator_probabilities = two_copy_hadamard_test(
        incoherent,
        include_z=True,
    )

    exact_purity = ancilla_expectation(
        purity_probabilities
    )
    exact_numerator = ancilla_expectation(
        numerator_probabilities
    )
    exact_ratio = exact_numerator / exact_purity

    print("\nTwo-copy Hadamard tests:")
    print(f"  Tr(rho^2):       {exact_purity:+.9f}")
    print(f"  Tr(Z rho^2):     {exact_numerator:+.9f}")
    print(f"  ratio:           {exact_ratio:+.9f}")

    rng = np.random.default_rng(20261001)
    raw_results = []
    distilled_results = []

    for _ in range(repetitions):
        raw_results.append(
            sample_raw_z(incoherent, shots, rng)
        )

        sampled_purity = sample_pm1(
            purity_probabilities,
            shots,
            rng,
        )
        sampled_numerator = sample_pm1(
            numerator_probabilities,
            shots,
            rng,
        )
        distilled_results.append(
            sampled_numerator / sampled_purity
        )

    raw_summary = summarize(raw_results, 1.0)
    distilled_summary = summarize(
        distilled_results,
        1.0,
    )

    ratio_predicted_variance = (
        (1.0 - exact_numerator**2)
        / (shots * exact_purity**2)
        + exact_numerator**2
        * (1.0 - exact_purity**2)
        / (shots * exact_purity**4)
    )

    print(
        f"\nFinite-shot results: {shots} shots per circuit, "
        f"{repetitions} repetitions"
    )
    print(" method       mean          bias          std         RMSE")
    print("-" * 67)
    for name, result in [
        ("raw", raw_summary),
        ("two-copy", distilled_summary),
    ]:
        print(
            f" {name:8s} {result[0]:+.9f}  {result[1]:+.3e}"
            f"  {result[2]:.6f}  {result[3]:.6f}"
        )

    print(
        "\nDelta-method prediction for two-copy std: "
        f"{np.sqrt(ratio_predicted_variance):.6f}"
    )
    print(
        f"Two-copy total circuit shots per estimate: "
        f"{2 * shots}"
    )

    coherent = coherent_rotation_state(coherent_angle)
    print("\nCoherent rotation error:")
    print(f"  Ry angle: {coherent_angle:.3f}")
    print(f"  raw <Z>:  {expectation_z(coherent):+.9f}")

    for copies in [2, 4, 8]:
        distilled = virtual_distilled_state(
            coherent,
            copies,
        )
        print(
            f"  M={copies} <Z>:  "
            f"{expectation_z(distilled):+.9f}"
        )


if __name__ == "__main__":
    main()
