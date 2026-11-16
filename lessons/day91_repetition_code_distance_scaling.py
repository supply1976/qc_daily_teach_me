"""Day 91: Repetition-code distance scaling and logical error suppression."""

from math import comb

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


def encoded_state(distance, theta=0.0, phase=0.0):
    """Encode one logical qubit as alpha|00...0> + beta|11...1>."""
    circuit = QuantumCircuit(distance)
    circuit.ry(float(theta), 0)
    circuit.rz(float(phase), 0)
    for qubit in range(1, distance):
        circuit.cx(0, qubit)
    return Statevector.from_instruction(circuit)


def syndrome(error_mask, distance):
    """Return adjacent Z_i Z_(i+1) parity checks for an X-error mask."""
    bits = [(error_mask >> qubit) & 1 for qubit in range(distance)]
    return tuple(bits[index] ^ bits[index + 1] for index in range(distance - 1))


def minimum_weight_correction(syndrome_bits):
    """Choose the lighter of the two error chains having this syndrome."""
    bits = [0]
    for parity in syndrome_bits:
        bits.append(bits[-1] ^ int(parity))

    first = sum(bit << qubit for qubit, bit in enumerate(bits))
    all_qubits = (1 << len(bits)) - 1
    complement = first ^ all_qubits

    if first.bit_count() <= complement.bit_count():
        return first
    return complement


def apply_x_mask(state, mask, distance):
    circuit = QuantumCircuit(distance)
    for qubit in range(distance):
        if (mask >> qubit) & 1:
            circuit.x(qubit)
    return state.evolve(circuit)


def recovered_fidelity(distance, error_mask):
    """Apply an X pattern, infer its minimum-weight chain, and recover."""
    logical_zero = encoded_state(distance)
    corrupted = apply_x_mask(logical_zero, error_mask, distance)
    correction = minimum_weight_correction(
        syndrome(error_mask, distance)
    )
    recovered = apply_x_mask(corrupted, correction, distance)
    return correction, float(state_fidelity(logical_zero, recovered))


def exact_logical_error_rate(distance, physical_error_rate):
    """Binomial tail: majority decoding fails above floor(distance / 2)."""
    first_failure_weight = distance // 2 + 1
    return sum(
        comb(distance, weight)
        * physical_error_rate**weight
        * (1.0 - physical_error_rate) ** (distance - weight)
        for weight in range(first_failure_weight, distance + 1)
    )


def enumerated_logical_error_rate(distance, physical_error_rate):
    """Enumerate all error masks and decode their stabilizer syndromes."""
    logical_error_rate = 0.0
    for error_mask in range(1 << distance):
        weight = error_mask.bit_count()
        probability = (
            physical_error_rate**weight
            * (1.0 - physical_error_rate) ** (distance - weight)
        )
        correction = minimum_weight_correction(
            syndrome(error_mask, distance)
        )
        residual = error_mask ^ correction
        if residual == (1 << distance) - 1:
            logical_error_rate += probability
    return logical_error_rate


def monte_carlo_logical_error_rate(
    distance,
    physical_error_rate,
    trials,
    rng,
):
    weights = rng.binomial(distance, physical_error_rate, size=trials)
    return float(np.mean(weights > distance // 2))


def main():
    physical_error_rate = 0.08
    trials = 1_000_000
    rng = np.random.default_rng(9105)

    print("Repetition-code distance scaling")
    print(f"physical X-error rate p: {physical_error_rate:.3f}")
    print(f"Monte Carlo trials:      {trials:,}")

    print("\nQiskit recovery check for distance d=5")
    print("weight   syndrome   correction   fidelity")
    print("-" * 48)
    distance = 5
    for weight in range(distance + 1):
        error_mask = (1 << weight) - 1
        checks = "".join(map(str, syndrome(error_mask, distance)))
        correction, fidelity = recovered_fidelity(distance, error_mask)
        print(
            f"{weight:6d}   {checks}"
            f"       {correction:0{distance}b}       {fidelity:.9f}"
        )

    print("\nLogical error suppression below threshold")
    print(" d   corrects t   exact P_L       Monte Carlo     P_L / p")
    print("-" * 66)
    distances = (1, 3, 5, 7, 9, 11)
    exact_rates = []
    for distance in distances:
        exact = exact_logical_error_rate(distance, physical_error_rate)
        sampled = monte_carlo_logical_error_rate(
            distance,
            physical_error_rate,
            trials,
            rng,
        )
        exact_rates.append(exact)
        print(
            f"{distance:2d}      {distance // 2:2d}"
            f"       {exact:.9e}   {sampled:.9e}   "
            f"{exact / physical_error_rate:.6f}"
        )

    print("\nExact formula versus explicit syndrome enumeration")
    for distance in (3, 5, 7):
        analytic = exact_logical_error_rate(distance, physical_error_rate)
        enumerated = enumerated_logical_error_rate(
            distance,
            physical_error_rate,
        )
        print(
            f"d={distance}: analytic={analytic:.12e}, "
            f"enumerated={enumerated:.12e}, "
            f"difference={abs(analytic - enumerated):.3e}"
        )

    print("\nCrossing the repetition-code threshold p=0.5")
    print(" d      P_L(p=0.08)      P_L(p=0.55)")
    print("-" * 44)
    for distance in distances:
        below = exact_logical_error_rate(distance, 0.08)
        above = exact_logical_error_rate(distance, 0.55)
        print(f"{distance:2d}    {below:.9f}       {above:.9f}")

    print("\nSmall-p leading term")
    distance = 7
    leading = (
        comb(distance, distance // 2 + 1)
        * physical_error_rate ** (distance // 2 + 1)
    )
    exact = exact_logical_error_rate(distance, physical_error_rate)
    print(f"d=7 exact P_L:         {exact:.12e}")
    print(f"leading 35 p^4 term:   {leading:.12e}")
    print(f"leading / exact:       {leading / exact:.9f}")


if __name__ == "__main__":
    main()
