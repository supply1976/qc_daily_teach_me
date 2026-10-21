"""Day 65: Classical shadows from randomized local Pauli measurements."""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


BASIS_NAMES = np.array(["X", "Y", "Z"])
BASIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def ghz_state():
    """Prepare (|000> + |111>) / sqrt(2) with Qiskit."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    return circuit, Statevector.from_instruction(circuit)


def measurement_probabilities(state, bases):
    """Probabilities after rotating local Pauli bases into Z."""
    rotation = QuantumCircuit(3)
    for qubit, basis in enumerate(bases):
        if basis == 0:  # X basis
            rotation.h(qubit)
        elif basis == 1:  # Y basis
            rotation.sdg(qubit)
            rotation.h(qubit)
    return state.evolve(rotation).probabilities()


def sample_local_pauli_shadows(state, shots, seed):
    """Return random bases and +/-1 outcomes for every qubit and shot."""
    rng = np.random.default_rng(seed)
    bases = rng.integers(0, 3, size=(shots, 3))
    outcomes = np.empty((shots, 3), dtype=np.int8)

    # Only 3^3 = 27 different basis circuits are needed.
    probability_cache = {
        basis_tuple: measurement_probabilities(state, basis_tuple)
        for basis_tuple in product(range(3), repeat=3)
    }

    for basis_tuple, probabilities in probability_cache.items():
        mask = np.all(bases == basis_tuple, axis=1)
        count = int(np.sum(mask))
        if count == 0:
            continue
        bitstring_indices = rng.choice(
            8,
            size=count,
            p=probabilities,
        )
        # Qiskit index bits are q2 q1 q0; extract outcome for each q.
        bits = np.column_stack([
            (bitstring_indices >> qubit) & 1
            for qubit in range(3)
        ])
        outcomes[mask] = 1 - 2 * bits

    return bases, outcomes


def pauli_labels(qubit_count):
    """All nonidentity Qiskit Pauli labels, written q_(n-1)...q_0."""
    labels = [
        "".join(characters)
        for characters in product("IXYZ", repeat=qubit_count)
    ]
    return [label for label in labels if set(label) != {"I"}]


def shadow_samples_for_pauli(label, bases, outcomes):
    """Single-shot unbiased estimators for one Pauli string."""
    estimates = np.ones(bases.shape[0])
    for qubit in range(bases.shape[1]):
        pauli = label[-1 - qubit]
        if pauli == "I":
            continue
        matches = bases[:, qubit] == BASIS_INDEX[pauli]
        estimates *= np.where(
            matches,
            3.0 * outcomes[:, qubit],
            0.0,
        )
    return estimates


def median_of_means(samples, group_count):
    if samples.size % group_count != 0:
        raise ValueError("shots must be divisible by group_count")
    group_means = samples.reshape(group_count, -1).mean(axis=1)
    return float(np.median(group_means))


def pauli_weight(label):
    return len(label) - label.count("I")


def exact_expectation(state, label):
    return float(np.real(state.expectation_value(Pauli(label))))


def main():
    shots = 30_000
    group_count = 15
    seed = 20261021

    circuit, state = ghz_state()
    bases, outcomes = sample_local_pauli_shadows(
        state,
        shots,
        seed,
    )

    labels = pauli_labels(3)
    results = {}
    for label in labels:
        single_shot_estimates = shadow_samples_for_pauli(
            label,
            bases,
            outcomes,
        )
        results[label] = {
            "exact": exact_expectation(state, label),
            "mean": float(np.mean(single_shot_estimates)),
            "median_of_means": median_of_means(
                single_shot_estimates,
                group_count,
            ),
            "second_moment": float(np.mean(single_shot_estimates**2)),
        }

    print("Classical shadows of a three-qubit GHZ state")
    print(circuit.draw(output="text"))
    print(f"Random Pauli shots: {shots}")
    print(f"Observables reused from the same data: {len(labels)}")
    print(f"Median-of-means groups: {group_count}")
    print("Basis counts per qubit (X, Y, Z):")
    for qubit in range(3):
        counts = np.bincount(bases[:, qubit], minlength=3)
        print(f"q{qubit}: {counts}")

    selected_labels = [
        "ZZI",
        "ZIZ",
        "IZZ",
        "XXX",
        "XYY",
        "YXY",
        "YYX",
        "ZII",
        "XII",
    ]
    print("\nSelected observables")
    print("Pauli  weight    exact      mean       median-of-means")
    print("-" * 58)
    for label in selected_labels:
        item = results[label]
        print(
            f"{label:>5s}  {pauli_weight(label):6d}  "
            f"{item['exact']:+.6f}  "
            f"{item['mean']:+.6f}  "
            f"{item['median_of_means']:+.6f}"
        )

    print("\nError and variance versus Pauli weight")
    print(
        "weight  strings  mean RMSE  MoM RMSE  "
        "observed E[snapshot^2]  theory 3^weight"
    )
    print("-" * 81)
    for weight in (1, 2, 3):
        subset = [
            results[label]
            for label in labels
            if pauli_weight(label) == weight
        ]
        mean_errors = np.array([
            item["mean"] - item["exact"]
            for item in subset
        ])
        mom_errors = np.array([
            item["median_of_means"] - item["exact"]
            for item in subset
        ])
        observed_second_moment = np.mean([
            item["second_moment"]
            for item in subset
        ])
        print(
            f"{weight:6d}  {len(subset):7d}  "
            f"{np.sqrt(np.mean(mean_errors**2)):.6f}  "
            f"{np.sqrt(np.mean(mom_errors**2)):.6f}  "
            f"{observed_second_moment:22.6f}  "
            f"{3**weight:15d}"
        )

    stabilizers = ["ZZI", "ZIZ", "IZZ", "XXX"]
    shadow_energy = -sum(
        results[label]["median_of_means"]
        for label in stabilizers
    )
    exact_energy = -sum(
        results[label]["exact"]
        for label in stabilizers
    )
    print("\nGHZ parent-Hamiltonian estimate")
    print("H = -(ZZI + ZIZ + IZZ + XXX)")
    print(f"Exact energy:  {exact_energy:+.9f}")
    print(f"Shadow energy: {shadow_energy:+.9f}")

    all_mean_errors = np.array([
        results[label]["mean"] - results[label]["exact"]
        for label in labels
    ])
    all_mom_errors = np.array([
        results[label]["median_of_means"]
        - results[label]["exact"]
        for label in labels
    ])
    print("\nAll 63 nonidentity Pauli strings")
    print(
        "Mean-estimator RMSE: "
        f"{np.sqrt(np.mean(all_mean_errors**2)):.9f}"
    )
    print(
        "MoM-estimator RMSE:  "
        f"{np.sqrt(np.mean(all_mom_errors**2)):.9f}"
    )
    print(
        "Maximum mean error:  "
        f"{np.max(np.abs(all_mean_errors)):.9f}"
    )


if __name__ == "__main__":
    main()
