"""Day 80: The [[5,1,3]] perfect quantum error-correcting code."""

from math import comb

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector, state_fidelity


NUM_QUBITS = 5


def qiskit_label(q0_to_q4):
    """Convert a human q0...q4 Pauli string to Qiskit's q4...q0 order."""
    return q0_to_q4[::-1]


# Cyclic stabilizer generators written in human q0...q4 order.
STABILIZER_WORDS = (
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ",
)
STABILIZERS = tuple(
    Pauli(qiskit_label(word))
    for word in STABILIZER_WORDS
)

LOGICAL_X = Pauli("XXXXX")
LOGICAL_Z = Pauli("ZZZZZ")


def code_projector():
    """Project onto the simultaneous +1 eigenspace of all stabilizers."""
    dimension = 2**NUM_QUBITS
    identity = np.eye(dimension, dtype=complex)
    projector = identity

    for stabilizer in STABILIZERS:
        projector = projector @ (
            identity + stabilizer.to_matrix()
        ) / 2.0

    return projector


def logical_codewords():
    """Construct |0_L> by projection and |1_L> = X_L |0_L>."""
    computational_zero = np.zeros(2**NUM_QUBITS, dtype=complex)
    computational_zero[0] = 1.0

    logical_zero = code_projector() @ computational_zero
    logical_zero /= np.linalg.norm(logical_zero)

    logical_one = LOGICAL_X.to_matrix() @ logical_zero
    return Statevector(logical_zero), Statevector(logical_one)


def input_amplitudes(theta, phase):
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    circuit.rz(phase, 0)
    return Statevector.from_instruction(circuit).data


def encoded_state(theta, phase, logical_pauli=None):
    amplitudes = input_amplitudes(theta, phase)
    logical_zero, logical_one = logical_codewords()
    data = (
        amplitudes[0] * logical_zero.data
        + amplitudes[1] * logical_one.data
    )

    state = Statevector(data)
    if logical_pauli == "X":
        state = state.evolve(LOGICAL_X)
    elif logical_pauli == "Z":
        state = state.evolve(LOGICAL_Z)
    elif logical_pauli == "Y":
        state = state.evolve(Pauli("YYYYY"))
    return state


def single_qubit_pauli(pauli, qubit):
    letters = ["I"] * NUM_QUBITS
    letters[qubit] = pauli
    return Pauli("".join(reversed(letters)))


def apply_errors(state, errors):
    damaged = state
    for pauli, qubit in errors:
        damaged = damaged.evolve(single_qubit_pauli(pauli, qubit))
    return damaged


def stabilizer_syndrome(state):
    bits = []
    expectations = []
    for stabilizer in STABILIZERS:
        value = float(np.real(state.expectation_value(stabilizer)))
        expectations.append(value)
        bits.append(int(np.rint((1.0 - value) / 2.0)))
    return tuple(bits), tuple(expectations)


def syndrome_text(syndrome):
    return "".join(str(bit) for bit in syndrome)


def build_decoder():
    """Assign each nonzero 4-bit syndrome to one single-qubit error."""
    reference = encoded_state(0.0, 0.0)
    decoder = {(0, 0, 0, 0): ("I", 0)}

    for qubit in range(NUM_QUBITS):
        for pauli in ("X", "Y", "Z"):
            damaged = apply_errors(reference, ((pauli, qubit),))
            syndrome, _ = stabilizer_syndrome(damaged)
            if syndrome in decoder:
                raise RuntimeError(
                    f"Syndrome collision for {pauli}{qubit}: {syndrome}"
                )
            decoder[syndrome] = (pauli, qubit)

    return decoder


def recover(state, decoder):
    syndrome, _ = stabilizer_syndrome(state)
    pauli, qubit = decoder[syndrome]
    if pauli == "I":
        return syndrome, (pauli, qubit), state
    corrected = state.evolve(single_qubit_pauli(pauli, qubit))
    return syndrome, (pauli, qubit), corrected


def quantum_hamming_bound(n, k, t):
    correctable_errors = sum(
        (3**weight) * comb(n, weight)
        for weight in range(t + 1)
    )
    occupied_dimension = (2**k) * correctable_errors
    hilbert_dimension = 2**n
    return correctable_errors, occupied_dimension, hilbert_dimension


def main():
    theta = 1.1
    phase = 0.7
    target = encoded_state(theta, phase)
    decoder = build_decoder()
    projector = code_projector()

    print("Five-qubit perfect code [[5,1,3]]")
    print(f"Input theta: {theta:.6f}")
    print(f"Input phase: {phase:.6f}")
    print(f"Code-space dimension Tr(P): {np.trace(projector).real:.1f}")

    print("\nStabilizer generators in human q0...q4 order")
    for index, word in enumerate(STABILIZER_WORDS, start=1):
        print(f"g{index} = {word}")

    errors, occupied, available = quantum_hamming_bound(
        n=5,
        k=1,
        t=1,
    )
    print("\nQuantum Hamming bound")
    print(f"Correctable Pauli patterns: {errors}")
    print(f"Occupied dimensions: 2^1 * {errors} = {occupied}")
    print(f"Available dimensions: 2^5 = {available}")
    print(f"Perfect packing equality: {occupied == available}")

    print("\nComplete syndrome table")
    print("error   syndrome   chosen correction   fidelity")
    print("-" * 56)
    fidelities = []
    observed_syndromes = set()
    for qubit in range(NUM_QUBITS):
        for pauli in ("X", "Y", "Z"):
            damaged = apply_errors(target, ((pauli, qubit),))
            syndrome, correction, corrected = recover(damaged, decoder)
            fidelity = float(state_fidelity(corrected, target))
            fidelities.append(fidelity)
            observed_syndromes.add(syndrome)
            print(
                f"{pauli}{qubit}      {syndrome_text(syndrome)}       "
                f"{correction[0]}{correction[1]}                 "
                f"{fidelity:.9f}"
            )

    print("\nExhaustive verification")
    print(f"Nonzero syndromes used: {len(observed_syndromes)} / 15")
    print(f"Errors tested: {len(fidelities)}")
    print(f"Minimum recovered fidelity: {min(fidelities):.12f}")
    print(
        "Maximum recovered infidelity: "
        f"{max(abs(1.0 - value) for value in fidelities):.3e}"
    )

    print("\nTwo-error example")
    double_error = (("X", 0), ("X", 1))
    damaged = apply_errors(target, double_error)
    syndrome, correction, corrected = recover(damaged, decoder)

    comparisons = {
        "I_L": target,
        "X_L": encoded_state(theta, phase, "X"),
        "Y_L": encoded_state(theta, phase, "Y"),
        "Z_L": encoded_state(theta, phase, "Z"),
    }
    print("Physical errors: X0 X1")
    print(f"Inferred syndrome: {syndrome_text(syndrome)}")
    print(f"Chosen correction: {correction[0]}{correction[1]}")
    for label, comparison in comparisons.items():
        fidelity = float(state_fidelity(corrected, comparison))
        print(f"F({label} target): {fidelity:.12f}")

    print("\nLogical-state stabilizer expectations")
    values = [
        float(np.real(target.expectation_value(stabilizer)))
        for stabilizer in STABILIZERS
    ]
    print(np.array2string(
        np.array(values),
        precision=1,
        floatmode="fixed",
    ))


if __name__ == "__main__":
    main()

