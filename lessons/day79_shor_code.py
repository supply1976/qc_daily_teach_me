"""Day 79: The nine-qubit Shor code corrects any single Pauli error."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Pauli,
    Statevector,
    partial_trace,
    state_fidelity,
)


NUM_QUBITS = 9


def pauli_label(pauli, qubits):
    """Return a Qiskit Pauli label; labels are ordered q8 ... q0."""
    letters = ["I"] * NUM_QUBITS
    for qubit in qubits:
        letters[qubit] = pauli
    return "".join(reversed(letters))


STABILIZER_LABELS = (
    pauli_label("Z", (0, 1)),
    pauli_label("Z", (1, 2)),
    pauli_label("Z", (3, 4)),
    pauli_label("Z", (4, 5)),
    pauli_label("Z", (6, 7)),
    pauli_label("Z", (7, 8)),
    pauli_label("X", (0, 1, 2, 3, 4, 5)),
    pauli_label("X", (3, 4, 5, 6, 7, 8)),
)


def prepare_logical_qubit(circuit, theta, phase):
    circuit.ry(theta, 0)
    circuit.rz(phase, 0)


def target_state(theta, phase, logical_pauli=None):
    circuit = QuantumCircuit(1)
    prepare_logical_qubit(circuit, theta, phase)
    if logical_pauli is not None:
        getattr(circuit, logical_pauli.lower())(0)
    return Statevector.from_instruction(circuit)


def shor_encoding_circuit():
    """Encode q0 into three GHZ blocks using q1...q8 as |0> ancillas."""
    circuit = QuantumCircuit(NUM_QUBITS, name="Shor encode")

    # Outer phase-flip repetition code: copy in the Z basis, then H.
    circuit.cx(0, 3)
    circuit.cx(0, 6)
    circuit.h(0)
    circuit.h(3)
    circuit.h(6)

    # Inner bit-flip repetition code in each three-qubit block.
    for leader in (0, 3, 6):
        circuit.cx(leader, leader + 1)
        circuit.cx(leader, leader + 2)

    return circuit


def encoded_state(theta, phase, errors=()):
    circuit = QuantumCircuit(NUM_QUBITS)
    prepare_logical_qubit(circuit, theta, phase)
    circuit.compose(shor_encoding_circuit(), inplace=True)

    for pauli, qubit in errors:
        getattr(circuit, pauli.lower())(qubit)

    return Statevector.from_instruction(circuit)


def stabilizer_syndrome(state):
    """Return 0 for +1 and 1 for -1 for all eight stabilizers."""
    bits = []
    expectations = []
    for label in STABILIZER_LABELS:
        value = float(np.real(
            state.expectation_value(Pauli(label))
        ))
        expectations.append(value)
        bits.append(int(np.rint((1.0 - value) / 2.0)))
    return tuple(bits), tuple(expectations)


def apply_pauli(state, pauli, qubit):
    circuit = QuantumCircuit(NUM_QUBITS)
    getattr(circuit, pauli.lower())(qubit)
    return state.evolve(circuit)


def build_single_error_decoder(theta=0.0, phase=0.0):
    """Map each single-error syndrome to one representative correction."""
    decoder = {(0,) * len(STABILIZER_LABELS): ("I", 0)}

    for qubit in range(NUM_QUBITS):
        for pauli in ("X", "Y", "Z"):
            state = encoded_state(
                theta,
                phase,
                errors=((pauli, qubit),),
            )
            syndrome, _ = stabilizer_syndrome(state)

            # Shor code is degenerate: distinct physical errors can have
            # the same syndrome but differ only by a stabilizer.
            decoder.setdefault(syndrome, (pauli, qubit))

    return decoder


def recover_and_decode(theta, phase, errors, decoder):
    damaged = encoded_state(theta, phase, errors)
    syndrome, _ = stabilizer_syndrome(damaged)
    correction_pauli, correction_qubit = decoder[syndrome]

    corrected = damaged
    if correction_pauli != "I":
        corrected = apply_pauli(
            corrected,
            correction_pauli,
            correction_qubit,
        )

    decoded = corrected.evolve(
        shor_encoding_circuit().inverse()
    )
    logical_density = partial_trace(
        decoded,
        list(range(1, NUM_QUBITS)),
    )
    return syndrome, (correction_pauli, correction_qubit), logical_density


def syndrome_text(syndrome):
    return "".join(str(bit) for bit in syndrome)


def main():
    theta = 1.1
    phase = 0.7
    target = target_state(theta, phase)
    logical_z_target = target_state(theta, phase, logical_pauli="Z")
    decoder = build_single_error_decoder()

    print("Nine-qubit Shor code")
    print(f"Input theta: {theta:.6f}")
    print(f"Input phase: {phase:.6f}")
    print(f"Number of stabilizers: {len(STABILIZER_LABELS)}")
    print(f"Distinct single-error syndromes: {len(decoder) - 1}")

    print("\nStabilizer generators in Qiskit q8...q0 order")
    for index, label in enumerate(STABILIZER_LABELS, start=1):
        print(f"S{index}: {label}")

    print("\nRepresentative errors")
    print("error   syndrome    chosen correction   recovered fidelity")
    print("-" * 67)
    for error in (("X", 4), ("Z", 4), ("Y", 4)):
        syndrome, correction, logical_density = recover_and_decode(
            theta,
            phase,
            (error,),
            decoder,
        )
        fidelity = float(state_fidelity(logical_density, target))
        print(
            f"{error[0]}{error[1]}      {syndrome_text(syndrome)}       "
            f"{correction[0]}{correction[1]:d}                 "
            f"{fidelity:.12f}"
        )

    print("\nExhaustive single-Pauli-error verification")
    fidelities = []
    failed_cases = []
    for qubit in range(NUM_QUBITS):
        for pauli in ("X", "Y", "Z"):
            syndrome, correction, logical_density = recover_and_decode(
                theta,
                phase,
                ((pauli, qubit),),
                decoder,
            )
            fidelity = float(state_fidelity(logical_density, target))
            fidelities.append(fidelity)
            if fidelity < 1.0 - 1e-10:
                failed_cases.append(
                    (pauli, qubit, syndrome, correction, fidelity)
                )

    print(f"Errors tested: {len(fidelities)}")
    print(f"Failed cases: {len(failed_cases)}")
    print(f"Minimum recovered fidelity: {min(fidelities):.12f}")
    print(
        "Maximum recovered infidelity: "
        f"{max(abs(1.0 - value) for value in fidelities):.3e}"
    )

    print("\nA two-error failure beyond code distance")
    double_error = (("X", 0), ("X", 1))
    syndrome, correction, logical_density = recover_and_decode(
        theta,
        phase,
        double_error,
        decoder,
    )
    original_fidelity = float(state_fidelity(logical_density, target))
    logical_z_fidelity = float(state_fidelity(
        logical_density,
        logical_z_target,
    ))
    print("Physical errors: X0 X1")
    print(f"Inferred syndrome: {syndrome_text(syndrome)}")
    print(f"Chosen correction: {correction[0]}{correction[1]}")
    print(f"F(original): {original_fidelity:.12f}")
    print(f"F(logical Z): {logical_z_fidelity:.12f}")

    print("\nLogical codeword stabilizer check")
    encoded = encoded_state(theta, phase)
    stabilizer_values = [
        float(np.real(encoded.expectation_value(Pauli(label))))
        for label in STABILIZER_LABELS
    ]
    print(np.array2string(
        np.array(stabilizer_values),
        precision=1,
        floatmode="fixed",
    ))


if __name__ == "__main__":
    main()

