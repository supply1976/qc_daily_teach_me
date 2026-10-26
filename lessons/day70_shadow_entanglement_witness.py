"""Day 70: Detect mixed-state entanglement with a shadow witness."""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector


PAULI_LABELS = ("X", "Y", "Z")


def werner_state(visibility):
    """Return v|Phi+><Phi+| + (1-v)I/4 and its preparation circuit."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    bell = Statevector.from_instruction(circuit)
    bell_density = DensityMatrix(bell).data
    matrix = (
        visibility * bell_density
        + (1.0 - visibility) * np.eye(4) / 4.0
    )
    return circuit, DensityMatrix(matrix)


def partial_transpose_q0(matrix):
    """Partially transpose q0 in Qiskit's |q1 q0> ordering."""
    tensor = np.asarray(matrix).reshape(2, 2, 2, 2)
    # Axes are (row q1, row q0, column q1, column q0).
    return tensor.transpose(0, 3, 2, 1).reshape(4, 4)


def exact_entanglement_metrics(density):
    partial_transpose = partial_transpose_q0(density.data)
    eigenvalues = np.linalg.eigvalsh(partial_transpose)
    negativity = float(np.sum(np.maximum(-eigenvalues, 0.0)))
    log_negativity = float(np.log2(1.0 + 2.0 * negativity))
    return eigenvalues, negativity, log_negativity


def measurement_probabilities(density, basis):
    rotation = QuantumCircuit(2)
    for qubit, basis_index in enumerate(basis):
        if basis_index == 0:  # X basis
            rotation.h(qubit)
        elif basis_index == 1:  # Y basis
            rotation.sdg(qubit)
            rotation.h(qubit)
    return density.evolve(rotation).probabilities()


def probability_cache(density):
    return {
        basis: measurement_probabilities(density, basis)
        for basis in product(range(3), repeat=2)
    }


def outcome_signs(indices):
    indices = np.asarray(indices)
    return np.column_stack([
        1 - 2 * (indices & 1),
        1 - 2 * ((indices >> 1) & 1),
    ])


def sample_shadow_correlations(cache, shots, rng):
    """Estimate XX, YY, ZZ from one uniform local-Pauli shadow."""
    bases = rng.integers(0, 3, size=(shots, 2))
    outcomes = np.empty((shots, 2), dtype=np.int8)

    for basis_tuple in product(range(3), repeat=2):
        mask = np.all(bases == basis_tuple, axis=1)
        count = int(np.sum(mask))
        if count == 0:
            continue
        sampled = rng.choice(4, size=count, p=cache[basis_tuple])
        outcomes[mask] = outcome_signs(sampled)

    estimates = {}
    compatible_counts = {}
    products = outcomes[:, 0] * outcomes[:, 1]
    for basis_index, label in enumerate(PAULI_LABELS):
        compatible = np.all(bases == basis_index, axis=1)
        # The compatible basis occurs with probability 1/9, hence weight 9.
        single_shot = np.where(compatible, 9.0 * products, 0.0)
        estimates[label + label] = float(np.mean(single_shot))
        compatible_counts[label + label] = int(np.sum(compatible))

    return estimates, compatible_counts


def metrics_from_correlations(correlations):
    """Bell witness and Werner-state negativity inferred from correlations."""
    fidelity = 0.25 * (
        1.0
        + correlations["XX"]
        - correlations["YY"]
        + correlations["ZZ"]
    )
    witness = 0.5 - fidelity
    raw_negativity = fidelity - 0.5
    clipped_negativity = max(0.0, raw_negativity)
    log_negativity = (
        np.log2(1.0 + 2.0 * clipped_negativity)
        if clipped_negativity > 0.0
        else 0.0
    )
    return {
        "fidelity": float(fidelity),
        "witness": float(witness),
        "raw_negativity": float(raw_negativity),
        "negativity": float(clipped_negativity),
        "log_negativity": float(log_negativity),
    }


def summary(values, truth):
    values = np.asarray(values)
    return {
        "mean": float(np.mean(values)),
        "bias": float(np.mean(values) - truth),
        "std": float(np.std(values, ddof=1)),
        "rmse": float(np.sqrt(np.mean((values - truth) ** 2))),
    }


def analytic_metrics(visibility):
    fidelity = (1.0 + 3.0 * visibility) / 4.0
    negativity = max(0.0, fidelity - 0.5)
    return fidelity, 0.5 - fidelity, negativity


def main():
    visibility = 0.45
    shots = 1_200
    trials = 500
    seed = 20261026

    circuit, density = werner_state(visibility)
    cache = probability_cache(density)
    pt_eigenvalues, exact_negativity, exact_log_negativity = (
        exact_entanglement_metrics(density)
    )
    exact_fidelity, exact_witness, analytic_negativity = (
        analytic_metrics(visibility)
    )

    results = {
        key: []
        for key in (
            "fidelity",
            "witness",
            "raw_negativity",
            "negativity",
            "log_negativity",
        )
    }
    first_correlations = None
    first_counts = None

    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        correlations, counts = sample_shadow_correlations(
            cache,
            shots,
            rng,
        )
        metrics = metrics_from_correlations(correlations)
        for key, value in metrics.items():
            results[key].append(value)
        if trial == 0:
            first_correlations = correlations
            first_counts = counts

    fidelity_interval = np.quantile(results["fidelity"], [0.05, 0.95])
    missed_fraction = float(np.mean(np.asarray(results["witness"]) >= 0.0))

    print("Mixed-state entanglement from a classical-shadow witness")
    print(circuit.draw(output="text"))
    print(f"Werner visibility v:             {visibility:.6f}")
    print(f"Shots per experiment:            {shots}")
    print(f"Independent experiments:         {trials}")
    print(f"Exact Bell fidelity:              {exact_fidelity:.9f}")
    print(f"Exact witness <W>:                {exact_witness:+.9f}")
    print(f"Partial-transpose eigenvalues:    {pt_eigenvalues}")
    print(f"Exact negativity:                 {exact_negativity:.9f}")
    print(f"Analytic negativity:              {analytic_negativity:.9f}")
    print(f"Exact logarithmic negativity:     {exact_log_negativity:.9f} bit")

    print("\nOne reusable shadow data set")
    print("observable   compatible shots   exact       estimate")
    print("-" * 59)
    exact_correlations = {"XX": visibility, "YY": -visibility, "ZZ": visibility}
    for label in ("XX", "YY", "ZZ"):
        print(
            f"{label:>8s}   {first_counts[label]:16d}   "
            f"{exact_correlations[label]:+.6f}   "
            f"{first_correlations[label]:+.6f}"
        )
    first_metrics = metrics_from_correlations(first_correlations)
    print(f"Bell fidelity estimate:           {first_metrics['fidelity']:.9f}")
    print(f"Witness estimate:                 {first_metrics['witness']:+.9f}")
    print(f"Clipped negativity estimate:      {first_metrics['negativity']:.9f}")

    print("\nMonte Carlo estimator behavior")
    print("quantity               truth        mean         bias         std          RMSE")
    print("-" * 86)
    truths = {
        "fidelity": exact_fidelity,
        "witness": exact_witness,
        "raw_negativity": exact_negativity,
        "negativity": exact_negativity,
        "log_negativity": exact_log_negativity,
    }
    display_names = {
        "fidelity": "Bell fidelity",
        "witness": "witness <W>",
        "raw_negativity": "raw F-1/2",
        "negativity": "clipped negativity",
        "log_negativity": "log negativity",
    }
    for key in results:
        stats = summary(results[key], truths[key])
        print(
            f"{display_names[key]:20s}  {truths[key]:+.7f}  "
            f"{stats['mean']:+.7f}  {stats['bias']:+.7f}  "
            f"{stats['std']:.7f}  {stats['rmse']:.7f}"
        )

    print(f"\nFidelity central 90% interval:    "
          f"[{fidelity_interval[0]:.9f}, {fidelity_interval[1]:.9f}]")
    print(f"Experiments missing entanglement: {100 * missed_fraction:.3f}%")

    print("\nWerner-state entanglement threshold")
    print("v          fidelity      witness       negativity")
    print("-" * 54)
    for value in (0.20, 1.0 / 3.0, 0.45, 0.80):
        fidelity, witness, negativity = analytic_metrics(value)
        print(
            f"{value:.6f}   {fidelity:.9f}   "
            f"{witness:+.9f}   {negativity:.9f}"
        )


if __name__ == "__main__":
    main()
