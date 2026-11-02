"""Day 77: Three-qubit bit-flip code and stabilizer syndromes."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Pauli,
    Statevector,
    partial_trace,
    state_fidelity,
)


def prepare_logical_qubit(circuit, theta, phase):
    circuit.ry(theta, 0)
    circuit.rz(phase, 0)


def target_state(theta, phase):
    circuit = QuantumCircuit(1)
    prepare_logical_qubit(circuit, theta, phase)
    return Statevector.from_instruction(circuit)


def encode_bit_flip_code(circuit):
    circuit.cx(0, 1)
    circuit.cx(0, 2)


def decode_bit_flip_code(circuit):
    circuit.cx(0, 2)
    circuit.cx(0, 1)


def extract_syndrome(circuit):
    """Store Z0Z1 parity in q3 and Z1Z2 parity in q4."""
    circuit.cx(0, 3)
    circuit.cx(1, 3)
    circuit.cx(1, 4)
    circuit.cx(2, 4)


def coherent_correction(circuit):
    """Correct X0, X1, or X2 from the two ancilla syndrome bits."""
    # syndrome q3 q4 = 10 -> error X0
    circuit.x(4)
    circuit.ccx(3, 4, 0)
    circuit.x(4)

    # syndrome q3 q4 = 11 -> error X1
    circuit.ccx(3, 4, 1)

    # syndrome q3 q4 = 01 -> error X2
    circuit.x(3)
    circuit.ccx(3, 4, 2)
    circuit.x(3)


def encoded_state(theta, phase, error_qubits=()):
    circuit = QuantumCircuit(3)
    prepare_logical_qubit(circuit, theta, phase)
    encode_bit_flip_code(circuit)
    for qubit in error_qubits:
        circuit.x(qubit)
    return Statevector.from_instruction(circuit)


def stabilizer_syndrome(theta, phase, error_qubits=()):
    state = encoded_state(theta, phase, error_qubits)

    # Pauli strings follow Qiskit's q2 q1 q0 order.
    z0_z1 = float(np.real(
        state.expectation_value(Pauli("IZZ"))
    ))
    z1_z2 = float(np.real(
        state.expectation_value(Pauli("ZZI"))
    ))
    syndrome = (
        int(np.rint((1.0 - z0_z1) / 2.0)),
        int(np.rint((1.0 - z1_z2) / 2.0)),
    )
    return z0_z1, z1_z2, syndrome


def recovered_logical_state(
    theta,
    phase,
    error_qubits=(),
    error_gate="x",
):
    circuit = QuantumCircuit(5)
    prepare_logical_qubit(circuit, theta, phase)
    encode_bit_flip_code(circuit)

    for qubit in error_qubits:
        getattr(circuit, error_gate)(qubit)

    extract_syndrome(circuit)
    coherent_correction(circuit)
    decode_bit_flip_code(circuit)

    final_state = Statevector.from_instruction(circuit)
    logical_density = partial_trace(
        final_state,
        [1, 2, 3, 4],
    )
    return circuit, logical_density


def bit_flipped_target(theta, phase):
    circuit = QuantumCircuit(1)
    prepare_logical_qubit(circuit, theta, phase)
    circuit.x(0)
    return Statevector.from_instruction(circuit)


def main():
    theta = 1.1
    phase = 0.7
    target = target_state(theta, phase)
    flipped_target = bit_flipped_target(theta, phase)

    print("Three-qubit bit-flip code with coherent syndrome recovery")
    print(f"Input theta: {theta:.6f}")
    print(f"Input phase: {phase:.6f}")

    print("\nSingle-error syndrome table")
    print("error   <Z0Z1>   <Z1Z2>   syndrome   recovered fidelity")
    print("-" * 66)
    single_cases = (
        ("none", ()),
        ("X0", (0,)),
        ("X1", (1,)),
        ("X2", (2,)),
    )
    maximum_single_error = 0.0
    for label, error_qubits in single_cases:
        first, second, syndrome = stabilizer_syndrome(
            theta,
            phase,
            error_qubits,
        )
        _, logical_density = recovered_logical_state(
            theta,
            phase,
            error_qubits,
        )
        fidelity = float(state_fidelity(
            logical_density,
            target,
        ))
        maximum_single_error = max(
            maximum_single_error,
            abs(1.0 - fidelity),
        )
        print(
            f"{label:>4s}    {first:+.1f}      {second:+.1f}       "
            f"{syndrome[0]}{syndrome[1]}        "
            f"{fidelity:.12f}"
        )

    print(
        "\nMaximum single-error infidelity: "
        f"{maximum_single_error:.3e}"
    )

    print("\nTwo simultaneous bit flips")
    print("errors   inferred syndrome   F(original)   F(logical X)")
    print("-" * 62)
    for error_qubits in ((0, 1), (0, 2), (1, 2)):
        _, _, syndrome = stabilizer_syndrome(
            theta,
            phase,
            error_qubits,
        )
        _, logical_density = recovered_logical_state(
            theta,
            phase,
            error_qubits,
        )
        original_fidelity = float(state_fidelity(
            logical_density,
            target,
        ))
        flipped_fidelity = float(state_fidelity(
            logical_density,
            flipped_target,
        ))
        label = "".join(str(qubit) for qubit in error_qubits)
        print(
            f" X{label}          {syndrome[0]}{syndrome[1]}          "
            f"{original_fidelity:.9f}    "
            f"{flipped_fidelity:.9f}"
        )

    print("\nExample recovery circuit for an X1 error")
    circuit, _ = recovered_logical_state(
        theta,
        phase,
        error_qubits=(1,),
    )
    print(circuit.draw(output="text"))

    # Demonstrate that the decoded logical state is pure after correction.
    _, corrected_density = recovered_logical_state(
        theta,
        phase,
        error_qubits=(2,),
    )
    purity = float(np.real(np.trace(
        corrected_density.data @ corrected_density.data
    )))
    print(f"Corrected logical-state purity: {purity:.12f}")


if __name__ == "__main__":
    main()

