"""Day 46: Pauli twirling and randomized compiling of coherent errors."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp


X = SparsePauliOp.from_list([("X", 1.0)])
Y = SparsePauliOp.from_list([("Y", 1.0)])
Z = SparsePauliOp.from_list([("Z", 1.0)])


def plus_state():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    return DensityMatrix.from_instruction(circuit)


def rz_evolve(density, angle):
    circuit = QuantumCircuit(1)
    circuit.rz(float(angle), 0)
    return density.evolve(circuit)


def coherent_state(depth, epsilon):
    """The same coherent Rz over-rotation occurs after every layer."""
    density = plus_state()
    for _ in range(depth):
        density = rz_evolve(density, epsilon)
    return density


def twirled_step(density, epsilon):
    """Average the +epsilon and -epsilon Pauli-frame branches."""
    plus = rz_evolve(density, epsilon)
    minus = rz_evolve(density, -epsilon)
    return DensityMatrix(0.5 * (plus.data + minus.data))


def twirled_state(depth, epsilon):
    density = plus_state()
    for _ in range(depth):
        density = twirled_step(density, epsilon)
    return density


def expectation(density, observable):
    return float(np.real(density.expectation_value(observable)))


def pauli_transfer_xy(epsilon, twirled):
    cosine = np.cos(epsilon)
    if twirled:
        return np.array([
            [cosine, 0.0],
            [0.0, cosine],
        ])

    sine = np.sin(epsilon)
    return np.array([
        [cosine, -sine],
        [sine, cosine],
    ])


def sample_x_from_angle(angle, shots, rng):
    probability_plus = 0.5 * (1.0 + np.cos(angle))
    plus_counts = rng.binomial(shots, probability_plus)
    return float((2 * plus_counts - shots) / shots)


def sample_randomized_compiling(depth, epsilon, shots, rng):
    """Draw an independent Pauli-frame sign for every layer and shot."""
    signs = rng.choice(
        [-1, 1],
        size=(shots, depth),
    )
    total_angles = epsilon * np.sum(signs, axis=1)
    probability_plus = 0.5 * (1.0 + np.cos(total_angles))
    outcomes = np.where(
        rng.random(shots) < probability_plus,
        1.0,
        -1.0,
    )
    return float(np.mean(outcomes))


def summarize(values, target):
    values = np.asarray(values)
    mean = float(np.mean(values))
    bias = mean - target
    standard_deviation = float(np.std(values, ddof=1))
    rmse = float(np.sqrt(np.mean((values - target) ** 2)))
    return mean, bias, standard_deviation, rmse


def main():
    epsilon = 0.08
    depths = [1, 5, 10, 20, 40]

    phase_flip_probability = np.sin(epsilon / 2.0) ** 2

    print("One-layer coherent Rz error:")
    print(f"  epsilon:                  {epsilon:.6f} rad")
    print(
        "  equivalent twirled Z rate: "
        f"{phase_flip_probability:.9f}"
    )

    print("\nPauli-transfer matrix on the X-Y plane:")
    print("coherent:")
    print(pauli_transfer_xy(epsilon, twirled=False))
    print("twirled:")
    print(pauli_transfer_xy(epsilon, twirled=True))

    print("\nDepth scaling from the |+> state:")
    print(" depth    coherent <X>   coherent <Y>    twirled <X>    twirled <Y>")
    print("-" * 78)

    maximum_error = 0.0
    for depth in depths:
        coherent = coherent_state(depth, epsilon)
        twirled = twirled_state(depth, epsilon)

        coherent_x = expectation(coherent, X)
        coherent_y = expectation(coherent, Y)
        twirled_x = expectation(twirled, X)
        twirled_y = expectation(twirled, Y)

        analytic = np.array([
            np.cos(depth * epsilon),
            np.sin(depth * epsilon),
            np.cos(epsilon) ** depth,
            0.0,
        ])
        simulated = np.array([
            coherent_x,
            coherent_y,
            twirled_x,
            twirled_y,
        ])
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(simulated - analytic))),
        )

        print(
            f" {depth:5d}    {coherent_x:+.9f}    {coherent_y:+.9f}"
            f"    {twirled_x:+.9f}    {twirled_y:+.9f}"
        )

    print(f"\nMaximum Qiskit-versus-analytic error: {maximum_error:.3e}")

    print("\nSmall-error prediction for loss 1-<X>:")
    print(" depth      coherent ~L^2 eps^2/2     twirled ~L eps^2/2")
    print("-" * 67)
    for depth in [1, 5, 10]:
        print(
            f" {depth:5d}          "
            f"{0.5 * depth**2 * epsilon**2:.9f}"
            f"             {0.5 * depth * epsilon**2:.9f}"
        )

    depth = 20
    shots = 4096
    repetitions = 500
    rng = np.random.default_rng(20261002)

    coherent_results = []
    randomized_results = []
    for _ in range(repetitions):
        coherent_results.append(
            sample_x_from_angle(
                depth * epsilon,
                shots,
                rng,
            )
        )
        randomized_results.append(
            sample_randomized_compiling(
                depth,
                epsilon,
                shots,
                rng,
            )
        )

    coherent_summary = summarize(coherent_results, 1.0)
    randomized_summary = summarize(randomized_results, 1.0)

    print(
        f"\nFinite-shot estimates at depth {depth}: "
        f"{shots} shots, {repetitions} repetitions"
    )
    print(" method          mean          bias          std         RMSE")
    print("-" * 70)
    for name, result in [
        ("coherent", coherent_summary),
        ("randomized", randomized_summary),
    ]:
        print(
            f" {name:10s}  {result[0]:+.9f}  {result[1]:+.3e}"
            f"  {result[2]:.6f}  {result[3]:.6f}"
        )

    print("\nExact ensemble expectations:")
    print(f"  coherent:  {np.cos(depth * epsilon):+.9f}")
    print(f"  randomized: {np.cos(epsilon) ** depth:+.9f}")


if __name__ == "__main__":
    main()
