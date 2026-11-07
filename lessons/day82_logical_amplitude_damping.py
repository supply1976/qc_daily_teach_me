"""Day 82: Logical amplitude damping and entanglement fidelity."""

from itertools import product

import numpy as np

from day80_five_qubit_perfect_code import (
    NUM_QUBITS,
    STABILIZERS,
    build_decoder,
    logical_codewords,
    single_qubit_pauli,
)


def amplitude_damping_kraus(gamma):
    k0 = np.array([
        [1.0, 0.0],
        [0.0, np.sqrt(1.0 - gamma)],
    ], dtype=complex)
    k1 = np.array([
        [0.0, np.sqrt(gamma)],
        [0.0, 0.0],
    ], dtype=complex)
    return k0, k1


def tensor_qiskit_order(single_qubit_operators):
    """Tensor q4 ... q0 so matrix indices match Qiskit's convention."""
    matrix = np.array([[1.0]], dtype=complex)
    for operator in reversed(single_qubit_operators):
        matrix = np.kron(matrix, operator)
    return matrix


def independent_amplitude_damping_kraus(gamma):
    """Return all 2^5 Kraus products for independent physical damping."""
    single_kraus = amplitude_damping_kraus(gamma)
    operators = []

    for branch in product((0, 1), repeat=NUM_QUBITS):
        factors = [single_kraus[index] for index in branch]
        operators.append(tensor_qiskit_order(factors))

    return operators


def syndrome_projector(syndrome):
    dimension = 2**NUM_QUBITS
    identity = np.eye(dimension, dtype=complex)
    projector = identity

    for bit, stabilizer in zip(syndrome, STABILIZERS):
        sign = 1.0 if bit == 0 else -1.0
        projector = projector @ (
            identity + sign * stabilizer.to_matrix()
        ) / 2.0

    return projector


def recovery_operators():
    """Construct U_s P_s for all 16 syndromes."""
    decoder = build_decoder()
    operators = []

    for syndrome, (pauli, qubit) in decoder.items():
        projector = syndrome_projector(syndrome)
        if pauli == "I":
            correction = np.eye(2**NUM_QUBITS, dtype=complex)
        else:
            correction = single_qubit_pauli(
                pauli,
                qubit,
            ).to_matrix()
        operators.append(correction @ projector)

    return operators


def encoded_bell_state():
    """Create (|0>|0_L> + |1>|1_L>) / sqrt(2)."""
    logical_zero, logical_one = logical_codewords()
    reference_zero = np.array([1.0, 0.0], dtype=complex)
    reference_one = np.array([0.0, 1.0], dtype=complex)

    state = (
        np.kron(reference_zero, logical_zero.data)
        + np.kron(reference_one, logical_one.data)
    ) / np.sqrt(2.0)
    return state


def apply_channel_to_system(density, kraus_operators):
    """Apply I_reference tensor E_system to a joint density matrix."""
    reference_identity = np.eye(2, dtype=complex)
    output = np.zeros_like(density)

    for operator in kraus_operators:
        joint_operator = np.kron(reference_identity, operator)
        output += (
            joint_operator
            @ density
            @ joint_operator.conj().T
        )

    return output


def pure_state_fidelity(density, target_state):
    return float(np.real(
        target_state.conj().T @ density @ target_state
    ))


def physical_entanglement_fidelity(gamma):
    """Analytic entanglement fidelity of one unencoded damped qubit."""
    return 0.25 * (1.0 + np.sqrt(1.0 - gamma)) ** 2


def logical_entanglement_fidelity(gamma, recovery):
    target = encoded_bell_state()
    initial_density = np.outer(target, target.conj())

    noisy_density = apply_channel_to_system(
        initial_density,
        independent_amplitude_damping_kraus(gamma),
    )
    recovered_density = apply_channel_to_system(
        noisy_density,
        recovery,
    )

    fidelity = pure_state_fidelity(recovered_density, target)
    trace = float(np.real(np.trace(recovered_density)))
    return fidelity, trace


def log_log_slope(gammas, infidelities):
    coefficients = np.polyfit(
        np.log(gammas),
        np.log(infidelities),
        deg=1,
    )
    return float(coefficients[0])


def main():
    recovery = recovery_operators()
    print("Five-qubit code under independent amplitude damping")
    print(f"Physical qubits: {NUM_QUBITS}")
    print(f"Noise Kraus branches: {2**NUM_QUBITS}")
    print(f"Recovery syndrome branches: {len(recovery)}")

    print("\nEntanglement fidelity")
    print("gamma    physical F_e    logical F_e     improvement    trace")
    print("-" * 72)
    for gamma in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
        physical = physical_entanglement_fidelity(gamma)
        logical, trace = logical_entanglement_fidelity(
            gamma,
            recovery,
        )
        physical_error = 1.0 - physical
        logical_error = 1.0 - logical
        improvement = (
            np.inf
            if logical_error < 1e-15
            else physical_error / logical_error
        )
        improvement_text = (
            "inf" if np.isinf(improvement) else f"{improvement:10.3f}"
        )
        print(
            f"{gamma:5.2f}    {physical:.9f}    {logical:.9f}    "
            f"{improvement_text:>10s}    {trace:.9f}"
        )

    small_gammas = np.array([
        1e-4,
        2e-4,
        5e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
    ])
    physical_infidelities = np.array([
        1.0 - physical_entanglement_fidelity(gamma)
        for gamma in small_gammas
    ])
    logical_infidelities = np.array([
        1.0 - logical_entanglement_fidelity(gamma, recovery)[0]
        for gamma in small_gammas
    ])

    physical_slope = log_log_slope(
        small_gammas,
        physical_infidelities,
    )
    logical_slope = log_log_slope(
        small_gammas,
        logical_infidelities,
    )

    print("\nSmall-noise power-law fit: 1 - F_e approximately gamma^p")
    print(f"Physical exponent p: {physical_slope:.6f}")
    print(f"Logical exponent p:  {logical_slope:.6f}")

    gamma = 0.02
    physical_error = 1.0 - physical_entanglement_fidelity(gamma)
    logical_error = 1.0 - logical_entanglement_fidelity(
        gamma,
        recovery,
    )[0]
    print("\nCheckpoint at gamma = 0.02")
    print(f"Physical infidelity: {physical_error:.12e}")
    print(f"Logical infidelity:  {logical_error:.12e}")
    print(f"Suppression factor:  {physical_error / logical_error:.6f}")


if __name__ == "__main__":
    main()

