"""Day 68: Purity and Renyi-2 entropy from classical-shadow U-statistics."""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = np.array([X, Y, Z])


def depolarized_bell_state(noise_probability):
    """rho = (1-p)|Phi+><Phi+| + p I/4."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    bell = Statevector.from_instruction(circuit)
    pure_density = DensityMatrix(bell).data
    mixed_density = (
        (1.0 - noise_probability) * pure_density
        + noise_probability * np.eye(4) / 4.0
    )
    return circuit, DensityMatrix(mixed_density)


def measurement_probabilities(state, basis):
    """Rotate a local X/Y/Z product basis into computational Z."""
    rotation = QuantumCircuit(2)
    for qubit, basis_index in enumerate(basis):
        if basis_index == 0:  # X
            rotation.h(qubit)
        elif basis_index == 1:  # Y
            rotation.sdg(qubit)
            rotation.h(qubit)
    return state.evolve(rotation).probabilities()


def probability_cache(state):
    return {
        basis: measurement_probabilities(state, basis)
        for basis in product(range(3), repeat=2)
    }


def outcome_signs(indices):
    """Return +/-1 outcomes in q0, q1 order."""
    indices = np.asarray(indices)
    return np.column_stack([
        1 - 2 * (indices & 1),
        1 - 2 * ((indices >> 1) & 1),
    ])


def local_snapshot(basis_index, sign):
    """Inverse local-Pauli measurement channel: (I + 3 s P)/2."""
    return 0.5 * (I2 + 3.0 * sign * PAULIS[basis_index])


def sample_snapshots(cache, shots, rng):
    """Generate independent two-qubit classical-shadow matrices."""
    bases = rng.integers(0, 3, size=(shots, 2))
    outcomes = np.empty((shots, 2), dtype=np.int8)

    for basis_tuple in product(range(3), repeat=2):
        mask = np.all(bases == basis_tuple, axis=1)
        count = int(np.sum(mask))
        if count == 0:
            continue
        indices = rng.choice(4, size=count, p=cache[basis_tuple])
        outcomes[mask] = outcome_signs(indices)

    snapshots = np.empty((shots, 4, 4), dtype=complex)
    for shot in range(shots):
        q0_snapshot = local_snapshot(bases[shot, 0], outcomes[shot, 0])
        q1_snapshot = local_snapshot(bases[shot, 1], outcomes[shot, 1])
        # Qiskit matrix order is q1 tensor q0.
        snapshots[shot] = np.kron(q1_snapshot, q0_snapshot)

    return snapshots


def purity_estimators(snapshots):
    """Return biased plug-in and unbiased complete U-statistic estimates."""
    shot_count = snapshots.shape[0]
    mean_snapshot = np.mean(snapshots, axis=0)
    plug_in = float(np.real(np.trace(mean_snapshot @ mean_snapshot)))

    # Sum over all distinct ordered pairs in O(N), rather than O(N^2).
    total = np.sum(snapshots, axis=0)
    all_pairs = float(np.real(np.vdot(total, total)))
    self_pairs = float(np.sum([
        np.real(np.vdot(snapshot, snapshot))
        for snapshot in snapshots
    ]))
    u_statistic = (
        (all_pairs - self_pairs)
        / (shot_count * (shot_count - 1))
    )
    return plug_in, u_statistic, mean_snapshot


def exact_purity(state):
    return float(np.real(np.trace(state.data @ state.data)))


def renyi2_entropy(purity):
    return float(-np.log2(max(purity, 1e-15)))


def summarize(estimates, truth):
    estimates = np.asarray(estimates)
    return {
        "mean": float(np.mean(estimates)),
        "bias": float(np.mean(estimates) - truth),
        "std": float(np.std(estimates, ddof=1)),
        "rmse": float(np.sqrt(np.mean((estimates - truth) ** 2))),
    }


def main():
    depolarizing_probability = 0.35
    shots = 400
    trials = 400
    seed = 20261024

    circuit, state = depolarized_bell_state(depolarizing_probability)
    cache = probability_cache(state)
    purity = exact_purity(state)
    entropy = renyi2_entropy(purity)

    plug_in_estimates = []
    u_estimates = []
    minimum_eigenvalues = []
    first_snapshots = None

    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        snapshots = sample_snapshots(cache, shots, rng)
        plug_in, u_statistic, reconstructed = purity_estimators(snapshots)
        plug_in_estimates.append(plug_in)
        u_estimates.append(u_statistic)
        minimum_eigenvalues.append(
            float(np.min(np.linalg.eigvalsh(reconstructed)))
        )
        if trial == 0:
            first_snapshots = snapshots

    plug_in_summary = summarize(plug_in_estimates, purity)
    u_summary = summarize(u_estimates, purity)

    # For two local Pauli snapshots, Tr(snapshot^2) = 5^2 = 25.
    snapshot_second_moment = 25.0
    theoretical_plug_in_mean = (
        (shots - 1) * purity + snapshot_second_moment
    ) / shots

    first_plug_in, first_u, first_reconstruction = purity_estimators(
        first_snapshots
    )
    first_eigenvalues = np.linalg.eigvalsh(first_reconstruction)

    print("Classical-shadow purity from a complete U-statistic")
    print(circuit.draw(output="text"))
    print(f"Depolarizing probability: {depolarizing_probability:.3f}")
    print(f"Shots per experiment:     {shots}")
    print(f"Independent experiments:  {trials}")
    print(f"Exact purity Tr(rho^2):    {purity:.9f}")
    print(f"Exact Renyi-2 entropy:     {entropy:.9f} bit")
    print(f"Tr(single snapshot^2):     {snapshot_second_moment:.1f}")
    print(
        "Theoretical plug-in mean: "
        f"{theoretical_plug_in_mean:.9f}"
    )

    print("\nPurity estimates over independent experiments")
    print("estimator       mean        bias        std         RMSE")
    print("-" * 63)
    for name, summary in (
        ("plug-in", plug_in_summary),
        ("U-statistic", u_summary),
    ):
        print(
            f"{name:11s}  {summary['mean']:.9f}  "
            f"{summary['bias']:+.9f}  "
            f"{summary['std']:.9f}  "
            f"{summary['rmse']:.9f}"
        )

    print("\nOne reproducible experiment")
    print(f"Plug-in purity:             {first_plug_in:.9f}")
    print(f"U-statistic purity:         {first_u:.9f}")
    print(
        "U-statistic Renyi-2:       "
        f"{renyi2_entropy(first_u):.9f} bit"
    )
    print(
        "Reconstructed eigenvalues: "
        + np.array2string(first_eigenvalues, precision=9)
    )
    print(
        "Fraction of nonphysical linear reconstructions: "
        f"{np.mean(np.array(minimum_eigenvalues) < -1e-12):.3%}"
    )

    efficient_u = purity_estimators(first_snapshots)[1]
    pairwise_u = np.mean([
        np.real(np.trace(first_snapshots[left] @ first_snapshots[right]))
        for left in range(shots)
        for right in range(shots)
        if left != right
    ])
    print("\nO(N) identity check against explicit O(N^2) pairs")
    print(f"Efficient U-statistic: {efficient_u:.12f}")
    print(f"Explicit pair average: {pairwise_u:.12f}")
    print(f"Absolute difference:   {abs(efficient_u - pairwise_u):.3e}")


if __name__ == "__main__":
    main()
