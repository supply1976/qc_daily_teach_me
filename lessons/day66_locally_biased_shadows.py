"""Day 66: Locally biased classical shadows for Hamiltonian energy."""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


BASIS_NAMES = np.array(["X", "Y", "Z"])
BASIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def hamiltonian_terms(coupling, transverse_field):
    """Three-qubit open-chain transverse-field Ising Hamiltonian."""
    return [
        ("IZZ", -float(coupling)),
        ("ZZI", -float(coupling)),
        ("IIX", -float(transverse_field)),
        ("IXI", -float(transverse_field)),
        ("XII", -float(transverse_field)),
    ]


def hamiltonian_operator(terms):
    return SparsePauliOp.from_list(terms)


def ground_state(operator):
    """Return the lowest eigenvalue and its normalized eigenvector."""
    energies, vectors = np.linalg.eigh(operator.to_matrix())
    return float(energies[0]), vectors[:, 0]


def prepare_state(vector):
    """Place the exact ground-state vector in a Qiskit circuit."""
    circuit = QuantumCircuit(3)
    circuit.initialize(vector, range(3))
    return circuit, Statevector.from_instruction(circuit)


def measurement_probabilities(state, bases):
    """Outcome probabilities after rotating X/Y measurements into Z."""
    rotation = QuantumCircuit(3)
    for qubit, basis in enumerate(bases):
        if basis == BASIS_INDEX["X"]:
            rotation.h(qubit)
        elif basis == BASIS_INDEX["Y"]:
            rotation.sdg(qubit)
            rotation.h(qubit)
    return state.evolve(rotation).probabilities()


def probability_cache(state):
    """There are only 3^3 local Pauli-basis circuits."""
    return {
        bases: measurement_probabilities(state, bases)
        for bases in product(range(3), repeat=3)
    }


def outcome_signs(indices, qubit_count=3):
    """Convert Qiskit basis-state indices to local +/-1 outcomes."""
    indices = np.asarray(indices)
    return np.column_stack([
        1 - 2 * ((indices >> qubit) & 1)
        for qubit in range(qubit_count)
    ])


def energy_samples(terms, bases, outcomes, distribution):
    """Unbiased locally biased shadow energy for every shot."""
    estimates = np.zeros(bases.shape[0])
    for label, coefficient in terms:
        matches = np.ones(bases.shape[0], dtype=bool)
        signs = np.ones(bases.shape[0])
        inverse_probability = 1.0

        for qubit in range(bases.shape[1]):
            pauli = label[-1 - qubit]
            if pauli == "I":
                continue
            basis_index = BASIS_INDEX[pauli]
            basis_probability = distribution[qubit, basis_index]
            if basis_probability <= 0:
                raise ValueError(
                    f"Observable {label} uses a basis with zero probability"
                )
            matches &= bases[:, qubit] == basis_index
            signs *= outcomes[:, qubit]
            inverse_probability /= basis_probability

        estimates += coefficient * np.where(
            matches,
            inverse_probability * signs,
            0.0,
        )
    return estimates


def exact_single_shot_moments(terms, distribution, cache):
    """Enumerate every basis and outcome to obtain exact mean/variance."""
    mean = 0.0
    second_moment = 0.0

    for basis_tuple in product(range(3), repeat=3):
        basis_probability = float(np.prod([
            distribution[qubit, basis]
            for qubit, basis in enumerate(basis_tuple)
        ]))
        if basis_probability == 0:
            continue

        indices = np.arange(8)
        bases = np.tile(basis_tuple, (8, 1))
        outcomes = outcome_signs(indices)
        estimators = energy_samples(
            terms,
            bases,
            outcomes,
            distribution,
        )
        event_probabilities = basis_probability * cache[basis_tuple]
        mean += float(np.sum(event_probabilities * estimators))
        second_moment += float(
            np.sum(event_probabilities * estimators**2)
        )

    return mean, second_moment - mean**2


def sample_energy(terms, distribution, cache, shots, seed):
    """Sample randomized local measurements without requiring Aer."""
    rng = np.random.default_rng(seed)
    bases = np.column_stack([
        rng.choice(3, size=shots, p=distribution[qubit])
        for qubit in range(3)
    ])
    outcomes = np.empty((shots, 3), dtype=np.int8)

    for basis_tuple in product(range(3), repeat=3):
        mask = np.all(bases == basis_tuple, axis=1)
        count = int(np.sum(mask))
        if count == 0:
            continue
        sampled_indices = rng.choice(
            8,
            size=count,
            p=cache[basis_tuple],
        )
        outcomes[mask] = outcome_signs(sampled_indices)

    return energy_samples(terms, bases, outcomes, distribution)


def xz_distribution(x_probability):
    """Same local X/Z bias on every qubit; Y is never needed here."""
    return np.tile(
        [x_probability, 0.0, 1.0 - x_probability],
        (3, 1),
    )


def coefficient_proxy(x_probability, coupling, transverse_field):
    """State-independent diagonal upper-bound proxy for the variance."""
    z_probability = 1.0 - x_probability
    return (
        3.0 * transverse_field**2 / x_probability
        + 2.0 * coupling**2 / z_probability**2
    )


def main():
    coupling = 1.0
    transverse_field = 0.8
    shots = 10_000
    seed = 20261022

    terms = hamiltonian_terms(coupling, transverse_field)
    operator = hamiltonian_operator(terms)
    exact_energy, ground_vector = ground_state(operator)
    circuit, state = prepare_state(ground_vector)
    cache = probability_cache(state)

    x_grid = np.linspace(0.05, 0.95, 901)
    proxy_values = np.array([
        coefficient_proxy(x, coupling, transverse_field)
        for x in x_grid
    ])
    proxy_x = float(x_grid[np.argmin(proxy_values)])

    exact_variances = np.array([
        exact_single_shot_moments(
            terms,
            xz_distribution(x),
            cache,
        )[1]
        for x in x_grid
    ])
    oracle_x = float(x_grid[np.argmin(exact_variances)])

    protocols = {
        "uniform XYZ": np.full((3, 3), 1.0 / 3.0),
        "uniform XZ": xz_distribution(0.5),
        "coefficient biased": xz_distribution(proxy_x),
        "oracle benchmark": xz_distribution(oracle_x),
    }

    print("Locally biased classical shadows for Hamiltonian energy")
    print("H = -J(IZZ + ZZI) - h(IIX + IXI + XII)")
    print(f"J = {coupling:.3f}, h = {transverse_field:.3f}")
    print(f"Exact ground energy: {exact_energy:+.9f}")
    print(
        "Qiskit state-preparation check: "
        f"{np.real(state.expectation_value(operator)):+.9f}"
    )
    print(f"Circuit depth after initialize: {circuit.depth()}")
    print(f"Coefficient-proxy optimum p(X): {proxy_x:.3f}")
    print(f"Exact-variance oracle p(X):     {oracle_x:.3f}")
    print(f"Shots per protocol: {shots}")

    print("\nProtocol comparison")
    print(
        "protocol              pX      pY      pZ    "
        "single-shot var    estimate       error       theory SE"
    )
    print("-" * 103)
    for protocol_index, (name, distribution) in enumerate(
        protocols.items()
    ):
        exact_mean, exact_variance = exact_single_shot_moments(
            terms,
            distribution,
            cache,
        )
        samples = sample_energy(
            terms,
            distribution,
            cache,
            shots,
            seed + protocol_index,
        )
        estimate = float(np.mean(samples))
        error = estimate - exact_energy
        standard_error = np.sqrt(exact_variance / shots)
        probabilities = distribution[0]
        print(
            f"{name:20s}  "
            f"{probabilities[0]:.3f}   "
            f"{probabilities[1]:.3f}   "
            f"{probabilities[2]:.3f}      "
            f"{exact_variance:10.6f}  "
            f"{estimate:+.9f}  "
            f"{error:+.6f}  "
            f"{standard_error:.6f}"
        )
        if abs(exact_mean - exact_energy) > 1e-10:
            raise RuntimeError("The shadow estimator is not unbiased")

    uniform_variance = exact_single_shot_moments(
        terms,
        protocols["uniform XYZ"],
        cache,
    )[1]
    biased_variance = exact_single_shot_moments(
        terms,
        protocols["coefficient biased"],
        cache,
    )[1]
    print("\nResource interpretation")
    print(
        "Variance reduction versus uniform XYZ: "
        f"{uniform_variance / biased_variance:.3f}x"
    )
    print(
        "Shots needed for the same standard error: "
        f"{biased_variance / uniform_variance:.3%}"
    )
    print(
        "The coefficient rule needs no knowledge of the quantum state; "
        "the oracle is shown only as a benchmark."
    )


if __name__ == "__main__":
    main()
