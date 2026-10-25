"""Day 69: Subsystem purity and Renyi-2 entanglement from shadows."""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector, partial_trace


I2 = np.eye(2, dtype=complex)
PAULIS = np.array([
    Pauli("X").to_matrix(),
    Pauli("Y").to_matrix(),
    Pauli("Z").to_matrix(),
])


def generalized_ghz_state(theta, qubit_count=4):
    """Prepare cos(theta/2)|00...0> + sin(theta/2)|11...1>."""
    circuit = QuantumCircuit(qubit_count)
    circuit.ry(theta, 0)
    for qubit in range(qubit_count - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit, Statevector.from_instruction(circuit)


def measurement_probabilities(state, basis):
    rotation = QuantumCircuit(len(basis))
    for qubit, basis_index in enumerate(basis):
        if basis_index == 0:  # X
            rotation.h(qubit)
        elif basis_index == 1:  # Y
            rotation.sdg(qubit)
            rotation.h(qubit)
    return state.evolve(rotation).probabilities()


def probability_cache(state, qubit_count=4):
    return {
        basis: measurement_probabilities(state, basis)
        for basis in product(range(3), repeat=qubit_count)
    }


def outcome_signs(indices, qubit_count=4):
    indices = np.asarray(indices)
    return np.column_stack([
        1 - 2 * ((indices >> qubit) & 1)
        for qubit in range(qubit_count)
    ])


def sample_local_snapshots(cache, shots, qubit_count, rng):
    """Store each shot as a product of local 2x2 shadow matrices."""
    bases = rng.integers(0, 3, size=(shots, qubit_count))
    outcomes = np.empty((shots, qubit_count), dtype=np.int8)

    for basis_tuple in product(range(3), repeat=qubit_count):
        mask = np.all(bases == basis_tuple, axis=1)
        count = int(np.sum(mask))
        if count == 0:
            continue
        indices = rng.choice(
            2**qubit_count,
            size=count,
            p=cache[basis_tuple],
        )
        outcomes[mask] = outcome_signs(indices, qubit_count)

    local = 0.5 * (
        I2[None, None, :, :]
        + 3.0
        * outcomes[:, :, None, None]
        * PAULIS[bases]
    )
    return local


def subsystem_snapshots(local_snapshots, subsystem):
    """Tensor together only the local snapshots inside a subsystem."""
    ordered = sorted(subsystem, reverse=True)
    reduced = local_snapshots[:, ordered[0]]
    for qubit in ordered[1:]:
        local = local_snapshots[:, qubit]
        shots, old_dimension, _ = reduced.shape
        reduced = np.einsum(
            "nij,nkl->nikjl",
            reduced,
            local,
        ).reshape(shots, 2 * old_dimension, 2 * old_dimension)
    return reduced


def purity_u_statistic(snapshots):
    """Complete distinct-pair U-statistic, evaluated in O(N)."""
    shot_count = snapshots.shape[0]
    total = np.sum(snapshots, axis=0)
    all_pairs = float(np.real(np.vdot(total, total)))
    self_pairs = float(np.sum(
        np.real(np.einsum("nij,nij->n", snapshots.conj(), snapshots))
    ))
    return (
        (all_pairs - self_pairs)
        / (shot_count * (shot_count - 1))
    )


def purity(density_matrix):
    matrix = density_matrix.data
    return float(np.real(np.trace(matrix @ matrix)))


def renyi2(purity_value):
    return float(-np.log2(max(purity_value, 1e-15)))


def statistics(values, truth):
    values = np.asarray(values)
    return {
        "mean": float(np.mean(values)),
        "bias": float(np.mean(values) - truth),
        "std": float(np.std(values, ddof=1)),
        "rmse": float(np.sqrt(np.mean((values - truth) ** 2))),
    }


def main():
    theta = 1.1
    shots = 1_200
    trials = 300
    seed = 20261025
    qubit_count = 4

    circuit, state = generalized_ghz_state(theta, qubit_count)
    cache = probability_cache(state, qubit_count)

    subsystem_a = (0, 1)
    subsystem_b = (2, 3)
    rho_a = partial_trace(state, list(subsystem_b))
    rho_b = partial_trace(state, list(subsystem_a))
    exact_purity_a = purity(rho_a)
    exact_purity_b = purity(rho_b)
    exact_entropy = renyi2(exact_purity_a)
    analytic_purity = 1.0 - 0.5 * np.sin(theta) ** 2

    estimates_by_size = {1: [], 2: [], 3: []}
    estimates_a = []
    estimates_b = []
    estimates_symmetric = []
    entropy_symmetric = []
    first_result = None

    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        local = sample_local_snapshots(
            cache,
            shots,
            qubit_count,
            rng,
        )

        size_estimates = {}
        for size in (1, 2, 3):
            snapshots = subsystem_snapshots(
                local,
                tuple(range(size)),
            )
            size_estimates[size] = purity_u_statistic(snapshots)
            estimates_by_size[size].append(size_estimates[size])

        snapshots_b = subsystem_snapshots(local, subsystem_b)
        estimate_a = size_estimates[2]
        estimate_b = purity_u_statistic(snapshots_b)
        symmetric = 0.5 * (estimate_a + estimate_b)

        estimates_a.append(estimate_a)
        estimates_b.append(estimate_b)
        estimates_symmetric.append(symmetric)
        entropy_symmetric.append(renyi2(symmetric))

        if trial == 0:
            first_result = (
                size_estimates,
                estimate_a,
                estimate_b,
                symmetric,
                renyi2(symmetric),
            )

    print("Renyi-2 entanglement entropy from subsystem shadows")
    print(circuit.draw(output="text"))
    print(f"theta:                    {theta:.6f} rad")
    print(f"Schmidt probabilities:    {np.cos(theta / 2)**2:.9f}, "
          f"{np.sin(theta / 2)**2:.9f}")
    print(f"Shots per experiment:     {shots}")
    print(f"Independent experiments:  {trials}")
    print(f"Exact purity rho_A:        {exact_purity_a:.9f}")
    print(f"Exact purity rho_B:        {exact_purity_b:.9f}")
    print(f"Analytic purity:           {analytic_purity:.9f}")
    print(f"Exact Renyi-2 entropy:     {exact_entropy:.9f} bit")

    print("\nSame exact purity, increasing subsystem size")
    print("size  Tr(snapshot^2)   mean estimate     std          RMSE")
    print("-" * 67)
    for size in (1, 2, 3):
        summary = statistics(estimates_by_size[size], exact_purity_a)
        print(
            f"{size:4d}  {5**size:14d}   "
            f"{summary['mean']:.9f}  "
            f"{summary['std']:.9f}  "
            f"{summary['rmse']:.9f}"
        )

    print("\nPure-state A/B symmetry averaging")
    print("estimator       mean        bias        std         RMSE")
    print("-" * 63)
    for name, values in (
        ("rho_A", estimates_a),
        ("rho_B", estimates_b),
        ("(A+B)/2", estimates_symmetric),
    ):
        summary = statistics(values, exact_purity_a)
        print(
            f"{name:11s}  {summary['mean']:.9f}  "
            f"{summary['bias']:+.9f}  "
            f"{summary['std']:.9f}  "
            f"{summary['rmse']:.9f}"
        )

    entropy_summary = statistics(entropy_symmetric, exact_entropy)
    print("\nEntropy after the nonlinear -log2 transformation")
    print(f"Mean estimated entropy: {entropy_summary['mean']:.9f} bit")
    print(f"Entropy bias:           {entropy_summary['bias']:+.9f} bit")
    print(f"Entropy standard dev.:  {entropy_summary['std']:.9f} bit")

    sizes, first_a, first_b, first_symmetric, first_entropy = first_result
    print("\nOne reproducible experiment")
    for size in (1, 2, 3):
        print(f"Purity of q0..q{size - 1}: {sizes[size]:.9f}")
    print(f"Purity rho_A:            {first_a:.9f}")
    print(f"Purity rho_B:            {first_b:.9f}")
    print(f"Symmetry-averaged purity:{first_symmetric: .9f}")
    print(f"Estimated entanglement:  {first_entropy:.9f} bit")


if __name__ == "__main__":
    main()
