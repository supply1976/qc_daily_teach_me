"""Day 73: Entanglement swapping and repeater correlation decay."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    Statevector,
    partial_trace,
    state_fidelity,
)


def bell_states():
    """Return Phi+, Phi-, Psi+, Psi- in that order."""
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


def werner_bell_state(fidelity):
    """Bell-diagonal state with fidelity F to Phi+."""
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


def bell_measurement_circuit():
    """Rotate q1,q2 so computational measurement is a Bell measurement."""
    circuit = QuantumCircuit(4)
    circuit.cx(1, 2)
    circuit.h(1)
    return circuit


def branch_projector(q1_result, q2_result):
    """Project onto one computational outcome for the middle qubits."""
    diagonal = np.zeros(16)
    for basis_index in range(16):
        q1 = (basis_index >> 1) & 1
        q2 = (basis_index >> 2) & 1
        diagonal[basis_index] = float(
            q1 == q1_result and q2 == q2_result
        )
    return np.diag(diagonal)


def correction_circuit(q1_result, q2_result):
    """Use the Bell outcome to transform the endpoint pair to Phi+."""
    circuit = QuantumCircuit(4)
    if q2_result:
        circuit.x(3)
    if q1_result:
        circuit.z(3)
    return circuit


def bell_probabilities(density):
    return np.array([
        float(np.real(np.trace(density.data @ projector)))
        for projector in BELL_PROJECTORS
    ])


def swapping_branches(link_density):
    """Bell-measure the middle qubits of links (q0,q1) and (q2,q3)."""
    four_qubit_density = DensityMatrix(
        np.kron(link_density.data, link_density.data)
    )
    rotated = four_qubit_density.evolve(
        Operator(bell_measurement_circuit())
    ).data

    uncorrected_average = np.zeros((4, 4), dtype=complex)
    corrected_average = np.zeros((4, 4), dtype=complex)
    records = []

    for q1_result in (0, 1):
        for q2_result in (0, 1):
            projector = branch_projector(q1_result, q2_result)
            branch = projector @ rotated @ projector
            probability = float(np.real(np.trace(branch)))
            branch /= probability

            uncorrected_endpoint = partial_trace(
                DensityMatrix(branch),
                [1, 2],
            )
            correction = Operator(
                correction_circuit(q1_result, q2_result)
            )
            corrected_branch = DensityMatrix(branch).evolve(correction)
            corrected_endpoint = partial_trace(
                corrected_branch,
                [1, 2],
            )

            uncorrected_average += (
                probability * uncorrected_endpoint.data
            )
            corrected_average += probability * corrected_endpoint.data

            uncorrected_bell = int(np.argmax(
                bell_probabilities(uncorrected_endpoint)
            ))
            corrected_fidelity = float(state_fidelity(
                corrected_endpoint,
                BELL_STATES[0],
            ))
            records.append({
                "outcome": f"{q1_result}{q2_result}",
                "probability": probability,
                "uncorrected_bell": BELL_NAMES[uncorrected_bell],
                "corrected_fidelity": corrected_fidelity,
            })

    return (
        records,
        DensityMatrix(uncorrected_average),
        DensityMatrix(corrected_average),
    )


def fidelity_to_visibility(fidelity):
    """Isotropic two-qubit state: F=(1+3v)/4."""
    return (4.0 * fidelity - 1.0) / 3.0


def visibility_to_fidelity(visibility):
    return (1.0 + 3.0 * visibility) / 4.0


def analytic_swapped_fidelity(link_fidelity):
    visibility = fidelity_to_visibility(link_fidelity)
    return visibility_to_fidelity(visibility ** 2)


def repeated_swapping_fidelity(link_fidelity, level):
    """A balanced chain at level l contains 2**l elementary links."""
    visibility = fidelity_to_visibility(link_fidelity)
    link_count = 2 ** level
    return visibility_to_fidelity(visibility ** link_count)


def main():
    print("Entanglement swapping with Bell-measurement feed-forward")
    print(bell_measurement_circuit().draw(output="text"))

    pure_records, pure_uncorrected, pure_corrected = swapping_branches(
        werner_bell_state(1.0)
    )
    print("Pure elementary links")
    print("outcome   probability   endpoint before correction   corrected F")
    print("-" * 72)
    for record in pure_records:
        print(
            f"  {record['outcome']}      "
            f"{record['probability']:.6f}             "
            f"{record['uncorrected_bell']:4s}             "
            f"{record['corrected_fidelity']:.9f}"
        )
    print(
        "Ignored-outcome fidelity:  "
        f"{state_fidelity(pure_uncorrected, BELL_STATES[0]):.9f}"
    )
    print(
        "Feed-forward fidelity:     "
        f"{state_fidelity(pure_corrected, BELL_STATES[0]):.9f}"
    )

    link_fidelity = 0.9
    _, noisy_uncorrected, noisy_corrected = swapping_branches(
        werner_bell_state(link_fidelity)
    )
    qiskit_fidelity = float(state_fidelity(
        noisy_corrected,
        BELL_STATES[0],
    ))
    analytic_fidelity = analytic_swapped_fidelity(link_fidelity)

    print("\nNoisy Bell-diagonal links")
    print(f"Elementary-link fidelity:       {link_fidelity:.9f}")
    print(
        "Input visibility:              "
        f"{fidelity_to_visibility(link_fidelity):.9f}"
    )
    print(f"Qiskit swapped fidelity:        {qiskit_fidelity:.9f}")
    print(f"Analytic swapped fidelity:      {analytic_fidelity:.9f}")
    print(
        "Absolute analytic error:       "
        f"{abs(qiskit_fidelity - analytic_fidelity):.3e}"
    )
    print(
        "Ignored-outcome fidelity:      "
        f"{state_fidelity(noisy_uncorrected, BELL_STATES[0]):.9f}"
    )
    print("Corrected Bell probabilities:")
    for name, probability in zip(
        BELL_NAMES,
        bell_probabilities(noisy_corrected),
    ):
        print(f"  p({name}) = {probability:.9f}")

    shots = 4_096
    rng = np.random.default_rng(73)
    sampled = rng.multinomial(
        shots,
        bell_probabilities(noisy_corrected),
    )
    print(f"\nFinite Bell-basis sampling ({shots} shots)")
    print({
        name: int(count)
        for name, count in zip(BELL_NAMES, sampled)
    })
    print(f"Sampled swapped fidelity:       {sampled[0] / shots:.9f}")

    chain_fidelity = 0.95
    print("\nBalanced swapping chain without purification")
    print(f"Elementary-link fidelity: {chain_fidelity:.6f}")
    print("level   links   visibility          endpoint F    entangled?")
    print("-" * 68)
    for level in range(7):
        links = 2 ** level
        visibility = fidelity_to_visibility(chain_fidelity) ** links
        fidelity = visibility_to_fidelity(visibility)
        print(
            f"{level:5d}   {links:5d}   {visibility:.12f}   "
            f"{fidelity:.9f}      {fidelity > 0.5}"
        )


if __name__ == "__main__":
    main()
