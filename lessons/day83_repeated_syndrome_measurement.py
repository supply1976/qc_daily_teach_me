"""Day 83: Repeated syndrome measurements with noisy readout."""

from math import comb

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


ERROR_LABELS = ("I", "X0", "X1", "X2")

# Stabilizers Z0 Z1 and Z1 Z2 identify every correctable X error.
SYNDROMES = np.array([
    [0, 0],  # I
    [1, 0],  # X0 anticommutes only with Z0 Z1
    [1, 1],  # X1 anticommutes with both stabilizers
    [0, 1],  # X2 anticommutes only with Z1 Z2
], dtype=np.int8)

SYNDROME_TO_ERROR = {
    tuple(syndrome): index
    for index, syndrome in enumerate(SYNDROMES)
}


def encoded_state(theta=1.1, phase=0.7):
    """Encode an arbitrary qubit as alpha|000> + beta|111>."""
    circuit = QuantumCircuit(3)
    circuit.ry(theta, 0)
    circuit.rz(phase, 0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    return Statevector.from_instruction(circuit)


def apply_error(state, error_index):
    """Apply I, X0, X1, or X2 to a Statevector."""
    if error_index == 0:
        return state

    circuit = QuantumCircuit(3)
    circuit.x(error_index - 1)
    return state.evolve(circuit)


def recovery_fidelity(target, true_error, inferred_error):
    """Fidelity after the physical error and inferred correction."""
    damaged = apply_error(target, true_error)
    recovered = apply_error(damaged, inferred_error)
    return float(state_fidelity(target, recovered))


def majority_bit_error(readout_error, rounds):
    """Probability that odd-round majority vote returns one wrong bit."""
    threshold = rounds // 2 + 1
    return sum(
        comb(rounds, flips)
        * readout_error**flips
        * (1.0 - readout_error) ** (rounds - flips)
        for flips in range(threshold, rounds + 1)
    )


def analytic_syndrome_success(readout_error, rounds):
    """Both independent syndrome bits must have the correct majority."""
    bit_error = majority_bit_error(readout_error, rounds)
    return (1.0 - bit_error) ** 2


def repeated_syndrome_experiment(
    trials,
    data_error_probability,
    readout_error_probability,
    rounds,
    seed,
):
    """Monte Carlo experiment with one static data error per trial.

    The data error happens before the first round and remains fixed. Each
    stabilizer readout bit is independently flipped in every round.
    """
    rng = np.random.default_rng(seed)
    error_probabilities = np.array([
        1.0 - data_error_probability,
        data_error_probability / 3.0,
        data_error_probability / 3.0,
        data_error_probability / 3.0,
    ])
    true_errors = rng.choice(
        len(ERROR_LABELS),
        size=trials,
        p=error_probabilities,
    )
    true_syndromes = SYNDROMES[true_errors]

    readout_flips = (
        rng.random((trials, rounds, 2))
        < readout_error_probability
    )
    measured = np.bitwise_xor(
        true_syndromes[:, None, :],
        readout_flips,
    )
    voted_syndromes = (
        np.sum(measured, axis=1) > rounds // 2
    ).astype(np.int8)

    inferred_errors = np.array([
        SYNDROME_TO_ERROR[tuple(syndrome)]
        for syndrome in voted_syndromes
    ])
    success = inferred_errors == true_errors

    return {
        "success_probability": float(np.mean(success)),
        "logical_failure_probability": float(np.mean(~success)),
        "true_errors": true_errors,
        "inferred_errors": inferred_errors,
    }


def main():
    target = encoded_state()

    print("Three-qubit bit-flip code with noisy syndrome readout")
    print("Stabilizers: Z0 Z1 and Z1 Z2")
    print("Syndrome map:")
    for label, syndrome in zip(ERROR_LABELS, SYNDROMES):
        print(f"  {label:>2s} -> {syndrome[0]}{syndrome[1]}")

    print("\nQiskit recovery-fidelity matrix")
    print("rows = true error; columns = inferred correction")
    print("          " + "  ".join(f"{label:>3s}" for label in ERROR_LABELS))
    for true_error, label in enumerate(ERROR_LABELS):
        fidelities = [
            recovery_fidelity(target, true_error, inferred_error)
            for inferred_error in range(len(ERROR_LABELS))
        ]
        values = "  ".join(f"{value:3.1f}" for value in fidelities)
        print(f"{label:>3s}       {values}")

    trials = 200_000
    data_error_probability = 0.12
    readout_error_probability = 0.10

    print("\nRepeated noisy syndrome measurements")
    print(f"Trials:                 {trials}")
    print(f"Data-error probability: {data_error_probability:.3f}")
    print(f"Readout error per bit:  {readout_error_probability:.3f}")
    print("rounds   bit-majority error   analytic success   sampled success")
    print("-" * 68)

    for rounds in (1, 3, 5, 7):
        bit_error = majority_bit_error(
            readout_error_probability,
            rounds,
        )
        analytic = analytic_syndrome_success(
            readout_error_probability,
            rounds,
        )
        sampled = repeated_syndrome_experiment(
            trials=trials,
            data_error_probability=data_error_probability,
            readout_error_probability=readout_error_probability,
            rounds=rounds,
            seed=8300 + rounds,
        )["success_probability"]
        print(
            f"{rounds:6d}   {bit_error:18.9f}   "
            f"{analytic:16.9f}   {sampled:15.9f}"
        )

    print("\nMinimum odd rounds for analytic syndrome success >= 0.99")
    for rounds in range(1, 100, 2):
        success = analytic_syndrome_success(
            readout_error_probability,
            rounds,
        )
        if success >= 0.99:
            print(f"rounds = {rounds}, success = {success:.9f}")
            break

    print("\nImportant assumption")
    print(
        "The data error is static during all rounds. If new data errors "
        "occur between rounds, decode syndrome changes in space-time "
        "instead of taking a simple majority."
    )


if __name__ == "__main__":
    main()
