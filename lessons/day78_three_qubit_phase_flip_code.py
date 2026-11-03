"""Day 78: Three-qubit phase-flip code via a Hadamard basis change."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Pauli,
    Statevector,
    partial_trace,
    state_fidelity,
)


def prepare_logical_qubit(circuit, theta, phase):
    """Prepare an arbitrary one-qubit state on q0."""
    circuit.ry(theta, 0)
    circuit.rz(phase, 0)


def target_state(theta, phase):
    circuit = QuantumCircuit(1)
    prepare_logical_qubit(circuit, theta, phase)
    return Statevector.from_instruction(circuit)


def bit_flipped_target(theta, phase):
    circuit = QuantumCircuit(1)
    prepare_logical_qubit(circuit, theta, phase)
    circuit.x(0)
    return Statevector.from_instruction(circuit)


def encode_phase_flip_code(circuit):
    """Map alpha|0> + beta|1> to alpha|+++> + beta|--->."""
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    for qubit in range(3):
        circuit.h(qubit)


def rotate_to_repetition_basis(circuit):
    """Turn physical Z errors into X errors because H Z H = X."""
    for qubit in range(3):
        circuit.h(qubit)


def extract_bit_flip_syndrome(circuit):
    """Store Z0Z1 parity in q3 and Z1Z2 parity in q4."""
    circuit.cx(0, 3)
    circuit.cx(1, 3)
    circuit.cx(1, 4)
    circuit.cx(2, 4)


def coherent_bit_flip_correction(circuit):
    """Correct one X error in the rotated repetition basis."""
    # syndrome q3 q4 = 10 -> X0
    circuit.x(4)
    circuit.ccx(3, 4, 0)
    circuit.x(4)

    # syndrome q3 q4 = 11 -> X1
    circuit.ccx(3, 4, 1)

    # syndrome q3 q4 = 01 -> X2
    circuit.x(3)
    circuit.ccx(3, 4, 2)
    circuit.x(3)


def decode_repetition_code(circuit):
    circuit.cx(0, 2)
    circuit.cx(0, 1)


def encoded_state(theta, phase, error_qubits=()):
    circuit = QuantumCircuit(3)
    prepare_logical_qubit(circuit, theta, phase)
    encode_phase_flip_code(circuit)
    for qubit in error_qubits:
        circuit.z(qubit)
    return Statevector.from_instruction(circuit)


def phase_syndrome(theta, phase, error_qubits=()):
    """Read the eigenvalues of X0X1 and X1X2."""
    state = encoded_state(theta, phase, error_qubits)

    # Qiskit Pauli strings use q2 q1 q0 order.
    x0_x1 = float(np.real(
        state.expectation_value(Pauli("IXX"))
    ))
    x1_x2 = float(np.real(
        state.expectation_value(Pauli("XXI"))
    ))
    syndrome = (
        int(np.rint((1.0 - x0_x1) / 2.0)),
        int(np.rint((1.0 - x1_x2) / 2.0)),
    )
    return x0_x1, x1_x2, syndrome


def recovered_logical_state(theta, phase, error_qubits=()):
    circuit = QuantumCircuit(5)
    prepare_logical_qubit(circuit, theta, phase)
    encode_phase_flip_code(circuit)

    for qubit in error_qubits:
        circuit.z(qubit)

    # HZH = X: reuse the bit-flip syndrome and correction circuit.
    rotate_to_repetition_basis(circuit)
    extract_bit_flip_syndrome(circuit)
    coherent_bit_flip_correction(circuit)
    decode_repetition_code(circuit)

    final_state = Statevector.from_instruction(circuit)
    logical_density = partial_trace(
        final_state,
        [1, 2, 3, 4],
    )
    return circuit, logical_density


def main():
    theta = 1.1
    phase = 0.7
    target = target_state(theta, phase)
    flipped_target = bit_flipped_target(theta, phase)

    print("Three-qubit phase-flip code via H Z H = X")
    print(f"Input theta: {theta:.6f}")
    print(f"Input phase: {phase:.6f}")

    print("\nSingle phase-error syndrome table")
    print("error   <X0X1>   <X1X2>   syndrome   recovered fidelity")
    print("-" * 66)
    single_cases = (
        ("none", ()),
        ("Z0", (0,)),
        ("Z1", (1,)),
        ("Z2", (2,)),
    )
    maximum_single_error = 0.0
    for label, error_qubits in single_cases:
        first, second, syndrome = phase_syndrome(
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

    print("\nTwo simultaneous phase flips")
    print("errors   inferred syndrome   F(original)   F(logical X)")
    print("-" * 62)
    for error_qubits in ((0, 1), (0, 2), (1, 2)):
        _, _, syndrome = phase_syndrome(
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
            f" Z{label}          {syndrome[0]}{syndrome[1]}          "
            f"{original_fidelity:.9f}    "
            f"{flipped_fidelity:.9f}"
        )

    print("\nEncoded logical stabilizers")
    encoded = encoded_state(theta, phase)
    for label in ("IXX", "XXI"):
        expectation = encoded.expectation_value(Pauli(label))
        print(f"<{label}> = {float(np.real(expectation)):+.1f}")

    print("\nExample recovery circuit for a Z1 error")
    circuit, corrected_density = recovered_logical_state(
        theta,
        phase,
        error_qubits=(1,),
    )
    print(circuit.draw(output="text"))
    purity = float(np.real(np.trace(
        corrected_density.data @ corrected_density.data
    )))
    print(f"Corrected logical-state purity: {purity:.12f}")


if __name__ == "__main__":
    main()

