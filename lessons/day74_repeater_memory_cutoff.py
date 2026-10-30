"""Day 74: Heralded repeater links, memory decoherence, and cutoffs."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity


def bell_states():
    """Return Phi+, Phi-, Psi+, Psi- in Pauli-error order I, Z, X, XZ."""
    states = []
    final_gates = (
        (),
        (("z", 0),),
        (("x", 1),),
        (("x", 1), ("z", 0)),
    )
    for gates in final_gates:
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        for gate, qubit in gates:
            getattr(circuit, gate)(qubit)
        states.append(Statevector.from_instruction(circuit))
    return tuple(states)


BELL_NAMES = ("Phi+", "Phi-", "Psi+", "Psi-")
BELL_STATES = bell_states()
BELL_PROJECTORS = tuple(
    DensityMatrix(state).data
    for state in BELL_STATES
)
MAXIMALLY_MIXED = np.eye(4, dtype=complex) / 4.0


def werner_probabilities(fidelity):
    error = (1.0 - fidelity) / 3.0
    return np.array([fidelity, error, error, error])


def density_from_bell_probabilities(probabilities):
    matrix = sum(
        probability * projector
        for probability, projector in zip(
            probabilities,
            BELL_PROJECTORS,
        )
    )
    return DensityMatrix(matrix)


def werner_state(fidelity):
    return density_from_bell_probabilities(
        werner_probabilities(fidelity)
    )


def memory_depolarization(density, age, memory_lifetime):
    """Exponential decay toward I/4 after `age` clock cycles."""
    survival = np.exp(-age / memory_lifetime)
    matrix = (
        survival * density.data
        + (1.0 - survival) * MAXIMALLY_MIXED
    )
    return DensityMatrix(matrix)


def bell_probabilities(density):
    return np.array([
        float(np.real(np.trace(density.data @ projector)))
        for projector in BELL_PROJECTORS
    ])


def swap_bell_diagonal(first, second):
    """Ideal swapping: Pauli-frame error labels compose by XOR."""
    output = np.zeros(4)
    for first_label, first_probability in enumerate(first):
        for second_label, second_probability in enumerate(second):
            output[first_label ^ second_label] += (
                first_probability * second_probability
            )
    return output


def swapped_state_after_wait(initial_fidelity, age, memory_lifetime):
    fresh = werner_state(initial_fidelity)
    stored = memory_depolarization(
        fresh,
        age,
        memory_lifetime,
    )
    output_probabilities = swap_bell_diagonal(
        bell_probabilities(stored),
        bell_probabilities(fresh),
    )
    return stored, density_from_bell_probabilities(output_probabilities)


def endpoint_fidelity(initial_fidelity, age, memory_lifetime):
    """Closed form for the ideal swap of one stored and one fresh link."""
    visibility = (4.0 * initial_fidelity - 1.0) / 3.0
    memory_factor = np.exp(-age / memory_lifetime)
    return 0.25 + 0.75 * visibility**2 * memory_factor


def simulate_cutoff_policy(
    link_probability,
    initial_fidelity,
    memory_lifetime,
    cutoff,
    output_pairs,
    seed,
):
    """Simulate two heralded links; discard the first after `cutoff` cycles."""
    rng = np.random.default_rng(seed)
    waiting_ages = []
    total_cycles = 0
    discarded_pairs = 0

    while len(waiting_ages) < output_pairs:
        # Both links attempt in parallel until at least one succeeds.
        while True:
            total_cycles += 1
            successes = rng.random(2) < link_probability
            if np.any(successes):
                break

        if np.all(successes):
            waiting_ages.append(0)
            continue

        # Exactly one link is stored while the missing link keeps trying.
        age = 0
        while True:
            total_cycles += 1
            age += 1
            if rng.random() < link_probability:
                waiting_ages.append(age)
                break
            if cutoff is not None and age >= cutoff:
                discarded_pairs += 1
                break

    waiting_ages = np.asarray(waiting_ages, dtype=float)
    fidelities = endpoint_fidelity(
        initial_fidelity,
        waiting_ages,
        memory_lifetime,
    )
    return {
        "cutoff": cutoff,
        "mean_wait": float(np.mean(waiting_ages)),
        "mean_fidelity": float(np.mean(fidelities)),
        "cycles_per_pair": total_cycles / output_pairs,
        "rate": output_pairs / total_cycles,
        "discard_fraction": (
            discarded_pairs / (discarded_pairs + output_pairs)
        ),
    }


def exact_no_cutoff_statistics(
    link_probability,
    initial_fidelity,
    memory_lifetime,
):
    """Exact means for T1,T2 ~ Geometric(p), with support 1,2,..."""
    p = link_probability
    q = 1.0 - p
    memory_factor = np.exp(-1.0 / memory_lifetime)

    mean_delivery_time = (
        2.0 / p - 1.0 / (p * (2.0 - p))
    )
    mean_wait = 2.0 * q / (p * (2.0 - p))
    mean_decay = (
        p / (2.0 - p)
        * (1.0 + q * memory_factor)
        / (1.0 - q * memory_factor)
    )

    visibility = (4.0 * initial_fidelity - 1.0) / 3.0
    mean_fidelity = 0.25 + 0.75 * visibility**2 * mean_decay
    return mean_delivery_time, mean_wait, mean_fidelity


def main():
    link_probability = 0.10
    initial_fidelity = 0.99
    memory_lifetime = 15.0
    output_pairs = 100_000

    print("Heralded two-link repeater with memory cutoffs")
    print(f"Link success probability per cycle: {link_probability:.3f}")
    print(f"Fresh-link Bell fidelity:            {initial_fidelity:.6f}")
    print(f"Memory lifetime:                     {memory_lifetime:.1f} cycles")

    print("\nQiskit density-matrix check")
    print("age   stored-link F   swapped endpoint F   analytic F   error")
    print("-" * 72)
    for age in (0, 5, 10, 20):
        stored, endpoint = swapped_state_after_wait(
            initial_fidelity,
            age,
            memory_lifetime,
        )
        stored_fidelity = float(state_fidelity(
            stored,
            BELL_STATES[0],
        ))
        qiskit_fidelity = float(state_fidelity(
            endpoint,
            BELL_STATES[0],
        ))
        analytic = endpoint_fidelity(
            initial_fidelity,
            age,
            memory_lifetime,
        )
        print(
            f"{age:3d}   {stored_fidelity:.9f}      "
            f"{qiskit_fidelity:.9f}        "
            f"{analytic:.9f}   "
            f"{abs(qiskit_fidelity - analytic):.2e}"
        )

    exact_time, exact_wait, exact_fidelity = exact_no_cutoff_statistics(
        link_probability,
        initial_fidelity,
        memory_lifetime,
    )
    print("\nExact no-cutoff prediction")
    print(f"Mean delivery time: {exact_time:.9f} cycles")
    print(f"Mean memory wait:   {exact_wait:.9f} cycles")
    print(f"Mean endpoint F:    {exact_fidelity:.9f}")

    print(f"\nMonte Carlo policies ({output_pairs:,} delivered pairs each)")
    print(
        "cutoff   mean wait   mean F       cycles/pair   "
        "rate/cycle   discarded"
    )
    print("-" * 79)
    for index, cutoff in enumerate((1, 2, 5, 10, 20, None)):
        result = simulate_cutoff_policy(
            link_probability,
            initial_fidelity,
            memory_lifetime,
            cutoff,
            output_pairs,
            seed=74 + index,
        )
        cutoff_label = "none" if cutoff is None else str(cutoff)
        print(
            f"{cutoff_label:>6s}   "
            f"{result['mean_wait']:9.6f}   "
            f"{result['mean_fidelity']:.9f}   "
            f"{result['cycles_per_pair']:11.6f}   "
            f"{result['rate']:.9f}   "
            f"{result['discard_fraction']:.3%}"
        )


if __name__ == "__main__":
    main()
