"""Day 42: zero-noise extrapolation with unitary folding and shot noise."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp


def bell_state_before_entangling_gate():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    return DensityMatrix.from_instruction(circuit)


def cnot_circuit():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    return circuit


def global_depolarizing_channel(density, error_probability):
    dimension = density.dim
    mixed = np.eye(dimension, dtype=complex) / dimension
    return DensityMatrix(
        (1.0 - error_probability) * density.data
        + error_probability * mixed
    )


def folded_noisy_bell_state(noise_scale, error_probability):
    """CNOT^(odd scale) is ideally one CNOT, but has more gate noise."""
    if noise_scale % 2 != 1:
        raise ValueError("noise_scale must be a positive odd integer")

    density = bell_state_before_entangling_gate()
    cnot = cnot_circuit()
    for _ in range(noise_scale):
        density = density.evolve(cnot)
        density = global_depolarizing_channel(
            density,
            error_probability,
        )
    return density


def exact_expectation(density, label="ZZ"):
    operator = SparsePauliOp.from_list([(label, 1.0)])
    return float(np.real(density.expectation_value(operator)))


def sampled_zz(density, shots, rng):
    probabilities = np.real(np.diag(density.data))
    outcomes = rng.choice(4, size=shots, p=probabilities)
    q0 = outcomes & 1
    q1 = (outcomes >> 1) & 1
    values = (1 - 2 * q0) * (1 - 2 * q1)
    return float(np.mean(values))


def richardson_coefficients(noise_scales):
    """Lagrange coefficients for evaluating a polynomial at lambda=0."""
    scales = np.asarray(noise_scales, dtype=float)
    coefficients = np.ones_like(scales)
    for index, scale in enumerate(scales):
        for other_index, other_scale in enumerate(scales):
            if index != other_index:
                coefficients[index] *= (
                    -other_scale / (scale - other_scale)
                )
    return coefficients


def extrapolate(estimates, coefficients):
    return float(np.dot(coefficients, estimates))


def main():
    error_probability = 0.06
    noise_scales = np.array([1, 3, 5])
    shots = 4096
    repeats = 500

    states = [
        folded_noisy_bell_state(scale, error_probability)
        for scale in noise_scales
    ]
    exact_noisy = np.array([
        exact_expectation(state) for state in states
    ])

    linear_coefficients = richardson_coefficients(noise_scales[:2])
    quadratic_coefficients = richardson_coefficients(noise_scales)

    print(f"Two-qubit depolarizing error per CNOT: {error_probability:.3f}")
    print("Ideal <ZZ>: 1.000000000")
    print("\nNoise-scaled circuits:")
    print(" scale   CNOT count   exact noisy <ZZ>   analytic (1-p)^scale")
    print("-" * 67)
    for scale, value in zip(noise_scales, exact_noisy):
        print(
            f"{scale:6d}   {scale:10d}      {value:.9f}"
            f"          {(1-error_probability)**scale:.9f}"
        )

    print("\nRichardson coefficients:")
    print("  linear scales [1,3]:   ", linear_coefficients)
    print("  quadratic scales [1,3,5]:", quadratic_coefficients)
    print("\nInfinite-shot extrapolation:")
    print(f"  raw scale-1: {exact_noisy[0]:.9f}")
    print(
        "  linear ZNE:  "
        f"{extrapolate(exact_noisy[:2], linear_coefficients):.9f}"
    )
    print(
        "  quadratic ZNE: "
        f"{extrapolate(exact_noisy, quadratic_coefficients):.9f}"
    )

    raw_results = []
    linear_results = []
    quadratic_results = []
    rng = np.random.default_rng(4200)
    for _ in range(repeats):
        estimates = np.array([
            sampled_zz(state, shots, rng) for state in states
        ])
        raw_results.append(estimates[0])
        linear_results.append(
            extrapolate(estimates[:2], linear_coefficients)
        )
        quadratic_results.append(
            extrapolate(estimates, quadratic_coefficients)
        )

    print("\nFinite-shot results over 500 repetitions:")
    print(" method          mean        bias         std       outside [-1,1]")
    print("-" * 72)
    for name, values in [
        ("raw", raw_results),
        ("linear ZNE", linear_results),
        ("quadratic ZNE", quadratic_results),
    ]:
        values = np.asarray(values)
        outside = np.mean(np.abs(values) > 1.0)
        print(
            f" {name:13s} {np.mean(values):+.9f}"
            f"  {np.mean(values)-1.0:+.3e}  {np.std(values, ddof=1):.6f}"
            f"       {100*outside:5.1f}%"
        )

    print("\nCoefficient L1 norms (sampling-overhead indicator):")
    print(f"  raw:           1.000")
    print(f"  linear ZNE:    {np.sum(np.abs(linear_coefficients)):.3f}")
    print(f"  quadratic ZNE: {np.sum(np.abs(quadratic_coefficients)):.3f}")


if __name__ == "__main__":
    main()
