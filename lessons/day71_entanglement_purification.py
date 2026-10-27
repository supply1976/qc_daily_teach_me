"""Day 71: BBPSSW-style recurrence entanglement purification."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    Statevector,
    partial_trace,
    state_fidelity,
)


BELL_LABELS = ("Phi+", "Phi-", "Psi+", "Psi-")


def bell_states():
    """Build the four Bell states in Qiskit's |q1 q0> ordering."""
    states = []
    gates = (
        (),
        (("z", 0),),
        (("x", 1),),
        (("x", 1), ("z", 0)),
    )
    for final_gates in gates:
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        for gate, qubit in final_gates:
            getattr(circuit, gate)(qubit)
        states.append(Statevector.from_instruction(circuit))
    return tuple(states)


BELL_STATES = bell_states()
BELL_PROJECTORS = tuple(
    DensityMatrix(state).data
    for state in BELL_STATES
)


def werner_bell_state(fidelity):
    """Bell-diagonal Werner state with target |Phi+> fidelity F."""
    error_probability = (1.0 - fidelity) / 3.0
    probabilities = np.array([
        fidelity,
        error_probability,
        error_probability,
        error_probability,
    ])
    matrix = sum(
        probability * projector
        for probability, projector in zip(
            probabilities,
            BELL_PROJECTORS,
        )
    )
    return DensityMatrix(matrix)


def bilateral_cnot_circuit():
    """Pair 1 is q0,q1; pair 2 is q2,q3."""
    circuit = QuantumCircuit(4)
    circuit.cx(0, 2)  # Alice: source pair -> target pair
    circuit.cx(1, 3)  # Bob:   source pair -> target pair
    return circuit


def equal_target_projector():
    """Project onto target-pair outcomes q2q3 = 00 or 11."""
    diagonal = np.zeros(16)
    for basis_index in range(16):
        q2 = (basis_index >> 2) & 1
        q3 = (basis_index >> 3) & 1
        diagonal[basis_index] = float(q2 == q3)
    return np.diag(diagonal)


def bell_probabilities(density):
    return np.array([
        float(np.real(np.trace(projector @ density.data)))
        for projector in BELL_PROJECTORS
    ])


def purification_round(density):
    """Consume two identical pairs and keep pair 1 if q2=q3."""
    # Four-qubit order is |q3 q2 q1 q0>, hence pair 2 tensor pair 1.
    two_copies = DensityMatrix(
        np.kron(density.data, density.data)
    )
    circuit = bilateral_cnot_circuit()
    evolved = two_copies.evolve(Operator(circuit))

    projector = equal_target_projector()
    accepted = projector @ evolved.data @ projector
    success_probability = float(np.real(np.trace(accepted)))
    accepted /= success_probability

    source_pair = partial_trace(
        DensityMatrix(accepted),
        [2, 3],
    )
    output_fidelity = float(
        state_fidelity(source_pair, BELL_STATES[0])
    )
    return source_pair, success_probability, output_fidelity, circuit


def analytic_recurrence(fidelity):
    """Werner-state BBPSSW recurrence and acceptance probability."""
    error_probability = (1.0 - fidelity) / 3.0
    numerator = fidelity**2 + error_probability**2
    success_probability = (
        fidelity**2
        + 2.0 * fidelity * error_probability
        + 5.0 * error_probability**2
    )
    return numerator / success_probability, success_probability


def simulate_finite_batch(
    input_pair_count,
    success_probability,
    output_bell_probabilities,
    rng,
):
    """Sample accepted pairs and their Bell labels for a finite batch."""
    attempts = input_pair_count // 2
    accepted = int(rng.binomial(attempts, success_probability))
    bell_counts = rng.multinomial(
        accepted,
        output_bell_probabilities,
    )
    empirical_fidelity = bell_counts[0] / accepted
    return attempts, accepted, bell_counts, empirical_fidelity


def main():
    initial_fidelity = 0.75
    rounds = 4
    input_pair_count = 10_000
    seed = 20261027

    density = werner_bell_state(initial_fidelity)
    output, success, qiskit_fidelity, circuit = purification_round(density)
    analytic_fidelity, analytic_success = analytic_recurrence(
        initial_fidelity
    )
    output_probabilities = bell_probabilities(output)

    print("Recurrence entanglement purification")
    print(circuit.draw(output="text"))
    print(f"Initial Bell fidelity:            {initial_fidelity:.9f}")
    print(f"Qiskit output fidelity:           {qiskit_fidelity:.9f}")
    print(f"Analytic output fidelity:         {analytic_fidelity:.9f}")
    print(f"Qiskit success probability:       {success:.9f}")
    print(f"Analytic success probability:     {analytic_success:.9f}")
    print(f"Maximum exact discrepancy:        "
          f"{max(abs(qiskit_fidelity - analytic_fidelity), abs(success - analytic_success)):.3e}")

    print("\nConditional output Bell probabilities before re-twirling")
    for label, probability in zip(BELL_LABELS, output_probabilities):
        print(f"p({label:4s}) = {probability:.9f}")

    rng = np.random.default_rng(seed)
    attempts, accepted, bell_counts, empirical_fidelity = (
        simulate_finite_batch(
            input_pair_count,
            success,
            output_probabilities,
            rng,
        )
    )
    print("\nOne finite batch")
    print(f"Input noisy pairs:                {input_pair_count}")
    print(f"Two-pair purification attempts:   {attempts}")
    print(f"Accepted output pairs:            {accepted}")
    print(f"Empirical acceptance rate:        {accepted / attempts:.9f}")
    print(f"Bell-label counts:                "
          f"{dict(zip(BELL_LABELS, bell_counts.tolist()))}")
    print(f"Empirical output fidelity:        {empirical_fidelity:.9f}")

    print("\nRepeated rounds with ideal re-twirling")
    print("round   input F      success P     output F     cumulative yield")
    print("-" * 70)
    fidelity = initial_fidelity
    cumulative_yield = 1.0
    print(f"{0:5d}   {fidelity:.9f}         ---           ---      {cumulative_yield:.9f}")

    for round_index in range(1, rounds + 1):
        density = werner_bell_state(fidelity)
        _, qiskit_success, qiskit_output, _ = purification_round(density)
        analytic_output, analytic_success = analytic_recurrence(fidelity)

        if not np.allclose(
            [qiskit_output, qiskit_success],
            [analytic_output, analytic_success],
            atol=1e-12,
        ):
            raise RuntimeError("Qiskit and recurrence formula disagree")

        # Every attempt consumes two pairs, and succeeds with probability P.
        cumulative_yield *= analytic_success / 2.0
        print(
            f"{round_index:5d}   {fidelity:.9f}   "
            f"{analytic_success:.9f}   {analytic_output:.9f}   "
            f"{cumulative_yield:.9f}"
        )
        fidelity = analytic_output

    print("\nPurification threshold")
    print("input F      output F      change")
    print("-" * 42)
    for test_fidelity in (0.49, 0.50, 0.51, 0.75):
        next_fidelity, _ = analytic_recurrence(test_fidelity)
        print(
            f"{test_fidelity:.6f}   {next_fidelity:.9f}   "
            f"{next_fidelity - test_fidelity:+.9f}"
        )


if __name__ == "__main__":
    main()
