"""Day 67: Greedy derandomized classical-shadow measurement schedules."""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


BASIS_NAMES = np.array(["X", "Y", "Z"])
BASIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def perturbed_cluster_state(qubit_count=4):
    """Prepare a cluster state followed by known local perturbations."""
    circuit = QuantumCircuit(qubit_count)
    for qubit in range(qubit_count):
        circuit.h(qubit)
    for qubit in range(qubit_count - 1):
        circuit.cz(qubit, qubit + 1)
    reference_state = Statevector.from_instruction(circuit)

    y_angles = [0.23, -0.31, 0.41, -0.17]
    z_angles = [0.11, 0.19, -0.13, 0.29]
    for qubit in range(qubit_count):
        circuit.ry(y_angles[qubit], qubit)
        circuit.rz(z_angles[qubit], qubit)

    return circuit, Statevector.from_instruction(circuit), reference_state


def cluster_stabilizers(state, qubit_count=4):
    """Find all nonidentity Pauli strings with expectation +/-1."""
    labels = []
    expectations = []
    for characters in product("IXYZ", repeat=qubit_count):
        label = "".join(characters)
        if label == "I" * qubit_count:
            continue
        expectation = float(np.real(state.expectation_value(Pauli(label))))
        if abs(expectation) > 1.0 - 1e-10:
            labels.append(label)
            expectations.append(expectation)
    return labels, np.array(expectations)


def pauli_weight(label):
    return len(label) - label.count("I")


def basis_covers(label, basis):
    """Whether a full local Pauli basis can estimate this Pauli string."""
    for qubit in range(len(basis)):
        pauli = label[-1 - qubit]
        if pauli == "I":
            continue
        if basis[qubit] != BASIS_INDEX[pauli]:
            return False
    return True


def coverage_matrix(labels, candidates):
    return np.array([
        [basis_covers(label, basis) for label in labels]
        for basis in candidates
    ], dtype=bool)


def greedy_schedule(labels, shots):
    """Greedily cover the currently least-observed Pauli strings.

    The diminishing-return score is a simple pedagogical derandomization
    objective, not a verbatim implementation of the confidence-bound
    objective in Huang, Kueng, and Preskill (2021).
    """
    qubit_count = len(labels[0])
    candidates = np.array(
        list(product(range(3), repeat=qubit_count)),
        dtype=np.int8,
    )
    covers = coverage_matrix(labels, candidates)
    counts = np.zeros(len(labels), dtype=int)
    schedule = np.empty((shots, qubit_count), dtype=np.int8)

    for shot in range(shots):
        rewards = 1.0 / (counts + 1.0) ** 2
        scores = covers @ rewards
        choice = int(np.argmax(scores))
        schedule[shot] = candidates[choice]
        counts += covers[choice]

    return schedule, counts


def measurement_probabilities(state, basis):
    """Rotate a local Pauli product basis into the Z basis."""
    circuit = QuantumCircuit(len(basis))
    for qubit, basis_index in enumerate(basis):
        if basis_index == BASIS_INDEX["X"]:
            circuit.h(qubit)
        elif basis_index == BASIS_INDEX["Y"]:
            circuit.sdg(qubit)
            circuit.h(qubit)
    return state.evolve(circuit).probabilities()


def probability_cache(state, qubit_count=4):
    return {
        basis: measurement_probabilities(state, basis)
        for basis in product(range(3), repeat=qubit_count)
    }


def outcome_signs(indices, qubit_count=4):
    """Convert Qiskit basis-state indices into local +/-1 eigenvalues."""
    indices = np.asarray(indices)
    return np.column_stack([
        1 - 2 * ((indices >> qubit) & 1)
        for qubit in range(qubit_count)
    ])


def measure_schedule(schedule, cache, rng):
    """Sample all measurement outcomes, grouping identical circuits."""
    outcomes = np.empty_like(schedule, dtype=np.int8)
    for basis_tuple in product(range(3), repeat=schedule.shape[1]):
        mask = np.all(schedule == basis_tuple, axis=1)
        count = int(np.sum(mask))
        if count == 0:
            continue
        indices = rng.choice(
            2 ** schedule.shape[1],
            size=count,
            p=cache[basis_tuple],
        )
        outcomes[mask] = outcome_signs(indices, schedule.shape[1])
    return outcomes


def estimate_observables(labels, schedule, outcomes):
    """Average each Pauli product over all compatible shots."""
    estimates = np.empty(len(labels))
    counts = np.empty(len(labels), dtype=int)

    for label_index, label in enumerate(labels):
        compatible = np.ones(schedule.shape[0], dtype=bool)
        products = np.ones(schedule.shape[0], dtype=float)
        for qubit in range(schedule.shape[1]):
            pauli = label[-1 - qubit]
            if pauli == "I":
                continue
            compatible &= schedule[:, qubit] == BASIS_INDEX[pauli]
            products *= outcomes[:, qubit]

        counts[label_index] = int(np.sum(compatible))
        estimates[label_index] = (
            float(np.mean(products[compatible]))
            if counts[label_index] > 0
            else 0.0
        )

    return estimates, counts


def randomized_schedule(shots, qubit_count, rng):
    return rng.integers(0, 3, size=(shots, qubit_count), dtype=np.int8)


def run_trials(
    labels,
    exact,
    fixed_schedule,
    cache,
    shots,
    trials,
    seed,
):
    """Compare uniform random schedules with one fixed greedy schedule."""
    random_rmse = []
    greedy_rmse = []
    random_max_error = []
    greedy_max_error = []
    random_missing = []

    for trial in range(trials):
        random_rng = np.random.default_rng(seed + 2 * trial)
        greedy_rng = np.random.default_rng(seed + 2 * trial + 1)

        random_basis = randomized_schedule(
            shots,
            fixed_schedule.shape[1],
            random_rng,
        )
        random_outcomes = measure_schedule(
            random_basis,
            cache,
            random_rng,
        )
        random_estimates, random_counts = estimate_observables(
            labels,
            random_basis,
            random_outcomes,
        )

        greedy_outcomes = measure_schedule(
            fixed_schedule,
            cache,
            greedy_rng,
        )
        greedy_estimates, _ = estimate_observables(
            labels,
            fixed_schedule,
            greedy_outcomes,
        )

        random_error = random_estimates - exact
        greedy_error = greedy_estimates - exact
        random_rmse.append(np.sqrt(np.mean(random_error**2)))
        greedy_rmse.append(np.sqrt(np.mean(greedy_error**2)))
        random_max_error.append(np.max(np.abs(random_error)))
        greedy_max_error.append(np.max(np.abs(greedy_error)))
        random_missing.append(np.sum(random_counts == 0))

    return {
        "random_rmse": np.array(random_rmse),
        "greedy_rmse": np.array(greedy_rmse),
        "random_max_error": np.array(random_max_error),
        "greedy_max_error": np.array(greedy_max_error),
        "random_missing": np.array(random_missing),
    }


def basis_string(basis):
    """Display in Qiskit's q_(n-1)...q_0 order."""
    return "".join(BASIS_NAMES[np.asarray(basis)[::-1]])


def main():
    shots = 300
    trials = 300
    seed = 20261023

    circuit, state, reference_state = perturbed_cluster_state()
    labels, _ = cluster_stabilizers(reference_state)
    exact = np.array([
        float(np.real(state.expectation_value(Pauli(label))))
        for label in labels
    ])
    schedule, designed_counts = greedy_schedule(labels, shots)
    cache = probability_cache(state)

    unique_bases, basis_counts = np.unique(
        schedule,
        axis=0,
        return_counts=True,
    )
    order = np.argsort(-basis_counts)

    results = run_trials(
        labels,
        exact,
        schedule,
        cache,
        shots,
        trials,
        seed,
    )

    print("Greedy derandomized shadows of a perturbed cluster state")
    print(circuit.draw(output="text"))
    print(f"Target stabilizer observables: {len(labels)}")
    print(f"Measurements per trial:       {shots}")
    print(f"Independent trials:           {trials}")

    print("\nTarget Pauli strings")
    print("Pauli  weight  exact  greedy compatible shots")
    print("-" * 51)
    for label, value, count in zip(labels, exact, designed_counts):
        print(
            f"{label:>5s}  {pauli_weight(label):6d}  "
            f"{value:+.6f}  {count:23d}"
        )

    print("\nGreedy measurement schedule")
    print(f"Distinct basis circuits: {len(unique_bases)}")
    print("basis  repetitions  number of targets covered")
    print("-" * 48)
    covers = coverage_matrix(labels, unique_bases)
    for index in order:
        print(
            f"{basis_string(unique_bases[index]):>5s}  "
            f"{basis_counts[index]:11d}  "
            f"{int(np.sum(covers[index])):25d}"
        )

    uniform_expected_counts = np.array([
        shots / (3 ** pauli_weight(label))
        for label in labels
    ])
    print("\nCoverage comparison")
    print(
        "Uniform random expected compatible shots "
        f"(min/median/max): {np.min(uniform_expected_counts):.3f} / "
        f"{np.median(uniform_expected_counts):.3f} / "
        f"{np.max(uniform_expected_counts):.3f}"
    )
    print(
        "Greedy fixed compatible shots "
        f"(min/median/max): {np.min(designed_counts)} / "
        f"{np.median(designed_counts):.1f} / "
        f"{np.max(designed_counts)}"
    )

    print("\nAccuracy over independent experiments")
    print("metric                         uniform random   greedy fixed")
    print("-" * 66)
    print(
        "mean observable RMSE            "
        f"{np.mean(results['random_rmse']):.6f}       "
        f"{np.mean(results['greedy_rmse']):.6f}"
    )
    print(
        "90% quantile of max error        "
        f"{np.quantile(results['random_max_error'], 0.9):.6f}       "
        f"{np.quantile(results['greedy_max_error'], 0.9):.6f}"
    )
    print(
        "trials missing >= 1 observable   "
        f"{np.mean(results['random_missing'] > 0):.3%}         "
        "0.000%"
    )
    print(
        "RMSE improvement factor:         "
        f"{np.mean(results['random_rmse']) / np.mean(results['greedy_rmse']):.3f}x"
    )

    first_rng = np.random.default_rng(seed + 999_999)
    first_outcomes = measure_schedule(schedule, cache, first_rng)
    first_estimates, _ = estimate_observables(
        labels,
        schedule,
        first_outcomes,
    )
    print("\nOne reproducible greedy-shadow experiment")
    print("Pauli   exact   estimate   absolute error")
    print("-" * 47)
    for label, value, estimate in zip(labels, exact, first_estimates):
        print(
            f"{label:>5s}  {value:+.3f}  {estimate:+.6f}  "
            f"{abs(estimate - value):.6f}"
        )


if __name__ == "__main__":
    main()
