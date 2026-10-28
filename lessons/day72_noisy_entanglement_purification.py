"""Day 72: Entanglement purification with imperfect local CNOT gates."""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    Pauli,
    Statevector,
    partial_trace,
    state_fidelity,
)


def bell_states():
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


BELL_STATES = bell_states()
BELL_PROJECTORS = tuple(
    DensityMatrix(state).data
    for state in BELL_STATES
)


def werner_bell_state(fidelity):
    error_probability = (1.0 - fidelity) / 3.0
    probabilities = (
        fidelity,
        error_probability,
        error_probability,
        error_probability,
    )
    return DensityMatrix(sum(
        probability * projector
        for probability, projector in zip(
            probabilities,
            BELL_PROJECTORS,
        )
    ))


def bilateral_cnot_circuit():
    circuit = QuantumCircuit(4)
    circuit.cx(0, 2)  # Alice's local CNOT
    circuit.cx(1, 3)  # Bob's local CNOT
    return circuit


def pauli_operator(paulis_by_qubit, qubit_count=4):
    """Construct a Pauli matrix using Qiskit's |q3 q2 q1 q0> order."""
    label = ["I"] * qubit_count
    for qubit, pauli in paulis_by_qubit.items():
        label[qubit_count - 1 - qubit] = pauli
    return Pauli("".join(label)).to_matrix()


def two_qubit_error_operators(qubits):
    operators = []
    for first, second in product("IXYZ", repeat=2):
        if first == second == "I":
            continue
        operators.append(pauli_operator({
            qubits[0]: first,
            qubits[1]: second,
        }))
    return tuple(operators)


ALICE_ERRORS = two_qubit_error_operators((0, 2))
BOB_ERRORS = two_qubit_error_operators((1, 3))


def apply_two_qubit_depolarizing(matrix, error_probability, operators):
    """Apply I with 1-p, or one of 15 nonidentity Paulis with p/15."""
    noisy = (1.0 - error_probability) * matrix
    for operator in operators:
        noisy += (
            error_probability
            / len(operators)
            * operator
            @ matrix
            @ operator.conj().T
        )
    return noisy


def equal_target_projector():
    diagonal = np.zeros(16)
    for index in range(16):
        q2 = (index >> 2) & 1
        q3 = (index >> 3) & 1
        diagonal[index] = float(q2 == q3)
    return np.diag(diagonal)


TARGET_PARITY_PROJECTOR = equal_target_projector()


def noisy_purification_round(density, gate_error_probability):
    """One BBPSSW-style round with independent noisy local CNOTs."""
    two_copies = DensityMatrix(
        np.kron(density.data, density.data)
    )
    circuit = bilateral_cnot_circuit()
    evolved = two_copies.evolve(Operator(circuit)).data

    evolved = apply_two_qubit_depolarizing(
        evolved,
        gate_error_probability,
        ALICE_ERRORS,
    )
    evolved = apply_two_qubit_depolarizing(
        evolved,
        gate_error_probability,
        BOB_ERRORS,
    )

    accepted = (
        TARGET_PARITY_PROJECTOR
        @ evolved
        @ TARGET_PARITY_PROJECTOR
    )
    success_probability = float(np.real(np.trace(accepted)))
    accepted /= success_probability

    output = partial_trace(
        DensityMatrix(accepted),
        [2, 3],
    )
    fidelity = float(state_fidelity(output, BELL_STATES[0]))
    return output, success_probability, fidelity


def iterate_protocol(initial_fidelity, gate_error_probability, rounds):
    """Re-twirl to a Werner state after every successful round."""
    fidelity = initial_fidelity
    cumulative_yield = 1.0
    records = []

    for round_index in range(1, rounds + 1):
        density = werner_bell_state(fidelity)
        _, success, output_fidelity = noisy_purification_round(
            density,
            gate_error_probability,
        )
        cumulative_yield *= success / 2.0
        improvement = output_fidelity - fidelity
        records.append({
            "round": round_index,
            "input_fidelity": fidelity,
            "success": success,
            "output_fidelity": output_fidelity,
            "improvement": improvement,
            "yield": cumulative_yield,
        })
        fidelity = output_fidelity

    return records


def find_fixed_points(gate_error_probability):
    """Find approximate F'=F crossings on a dense one-dimensional grid."""
    grid = np.linspace(0.25, 0.9999, 1_001)
    changes = []
    for fidelity in grid:
        _, _, output = noisy_purification_round(
            werner_bell_state(fidelity),
            gate_error_probability,
        )
        changes.append(output - fidelity)
    changes = np.asarray(changes)

    roots = []
    for index in range(len(grid) - 1):
        if changes[index] * changes[index + 1] < 0.0:
            left = grid[index]
            right = grid[index + 1]
            left_change = changes[index]
            for _ in range(45):
                middle = 0.5 * (left + right)
                _, _, output = noisy_purification_round(
                    werner_bell_state(middle),
                    gate_error_probability,
                )
                change = output - middle
                if left_change * change <= 0.0:
                    right = middle
                else:
                    left = middle
                    left_change = change
            roots.append(0.5 * (left + right))
    return roots


def main():
    initial_fidelity = 0.75
    gate_error_probability = 0.02
    rounds = 20

    print("Entanglement purification with noisy local CNOTs")
    print(bilateral_cnot_circuit().draw(output="text"))
    print(f"Initial fidelity:                 {initial_fidelity:.6f}")
    print(f"Two-qubit Pauli error per CNOT:   {gate_error_probability:.3%}")
    print(f"Nonidentity errors per channel:   {len(ALICE_ERRORS)}")

    ideal = iterate_protocol(initial_fidelity, 0.0, rounds)
    noisy = iterate_protocol(
        initial_fidelity,
        gate_error_probability,
        rounds,
    )

    print("\nIdeal versus noisy purification")
    print("round   ideal F      noisy F      noisy gain    noisy success   noisy yield")
    print("-" * 85)
    for ideal_record, noisy_record in zip(ideal, noisy):
        print(
            f"{noisy_record['round']:5d}   "
            f"{ideal_record['output_fidelity']:.9f}  "
            f"{noisy_record['output_fidelity']:.9f}  "
            f"{noisy_record['improvement']:+.9f}  "
            f"{noisy_record['success']:.9f}  "
            f"{noisy_record['yield']:.9e}"
        )

    print("\nNoise-dependent purification window")
    print("gate error     lower threshold     upper ceiling")
    print("-" * 56)
    for error_probability in (0.0, 0.005, 0.01, 0.02, 0.05):
        roots = find_fixed_points(error_probability)
        if not roots:
            print(
                f"{error_probability:9.3%}     "
                "no improving interval"
            )
        else:
            lower = roots[0]
            upper = roots[-1] if len(roots) > 1 else 1.0
            print(
                f"{error_probability:9.3%}     "
                f"{lower:.9f}         {upper:.9f}"
            )

    gain_tolerance = 1e-3
    practical_stop = next(
        (
            record for record in noisy
            if record["improvement"] < gain_tolerance
        ),
        noisy[-1],
    )
    print("\nPractical stopping rule")
    print(f"Fidelity-gain tolerance:          {gain_tolerance:.1e}")
    print(f"First round below tolerance:      {practical_stop['round']}")
    print(f"Fidelity at that round:           {practical_stop['output_fidelity']:.9f}")
    print(f"Gain during that round:           {practical_stop['improvement']:.9f}")
    print(f"Yield at that round:              {practical_stop['yield']:.9e}")


if __name__ == "__main__":
    main()
