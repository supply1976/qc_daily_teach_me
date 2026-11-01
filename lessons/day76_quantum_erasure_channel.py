"""Day 76: Quantum erasure channel, coherent information, and capacity."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, entropy, partial_trace


def bell_density():
    """Prepare |Phi+> between reference q0 and transmitted qubit q1."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return DensityMatrix.from_instruction(circuit)


def quantum_erasure_channel(bipartite_density, erasure_probability):
    """Apply an erasure channel to q1 and replace it by a qutrit output.

    Output basis is |reference, receiver>, with receiver labels 0, 1, E.
    The Qiskit input basis is |q1 q0>.
    """
    epsilon = float(erasure_probability)
    input_matrix = bipartite_density.data
    output = np.zeros((6, 6), dtype=complex)

    # Unerased branch: embed receiver |0>,|1> into a qutrit.
    for reference_row in range(2):
        for receiver_row in range(2):
            input_row = 2 * receiver_row + reference_row
            output_row = 3 * reference_row + receiver_row
            for reference_column in range(2):
                for receiver_column in range(2):
                    input_column = (
                        2 * receiver_column + reference_column
                    )
                    output_column = (
                        3 * reference_column + receiver_column
                    )
                    output[output_row, output_column] += (
                        (1.0 - epsilon)
                        * input_matrix[input_row, input_column]
                    )

    # Erased branch: receiver gets the orthogonal flag |E>.
    reference_density = partial_trace(
        bipartite_density,
        [1],
    ).data
    for reference_row in range(2):
        for reference_column in range(2):
            output_row = 3 * reference_row + 2
            output_column = 3 * reference_column + 2
            output[output_row, output_column] += (
                epsilon
                * reference_density[reference_row, reference_column]
            )

    return DensityMatrix(output)


def receiver_reduced_density(joint_density):
    """Trace out the two-dimensional reference from a 2 x 3 state."""
    matrix = joint_density.data
    receiver = np.zeros((3, 3), dtype=complex)
    for reference in range(2):
        start = 3 * reference
        receiver += matrix[
            start:start + 3,
            start:start + 3,
        ]
    return DensityMatrix(receiver)


def binary_entropy(probability):
    probability = float(probability)
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(
        -probability * np.log2(probability)
        -(1.0 - probability) * np.log2(1.0 - probability)
    )


def coherent_information_qiskit(erasure_probability):
    input_density = bell_density()
    joint_output = quantum_erasure_channel(
        input_density,
        erasure_probability,
    )
    receiver_output = receiver_reduced_density(joint_output)

    receiver_entropy = float(entropy(receiver_output, base=2))
    joint_entropy = float(entropy(joint_output, base=2))
    coherent_information = receiver_entropy - joint_entropy
    return {
        "receiver_entropy": receiver_entropy,
        "joint_entropy": joint_entropy,
        "coherent_information": coherent_information,
        "joint_eigenvalues": np.linalg.eigvalsh(joint_output.data),
        "receiver_eigenvalues": np.linalg.eigvalsh(
            receiver_output.data
        ),
        "trace": float(np.real(np.trace(joint_output.data))),
    }


def quantum_capacity(erasure_probability):
    """Unassisted quantum capacity in qubits per channel use."""
    return max(0.0, 1.0 - 2.0 * erasure_probability)


def finite_block_experiment(
    erasure_probability,
    block_size,
    trials,
    seed,
):
    rng = np.random.default_rng(seed)
    erased_counts = rng.binomial(
        block_size,
        erasure_probability,
        size=trials,
    )

    # Conditioned on the erasure flags, the coherent-information
    # contribution is +1 for each received qubit and -1 for each erasure.
    raw_rates = 1.0 - 2.0 * erased_counts / block_size
    return {
        "mean": float(np.mean(raw_rates)),
        "standard_deviation": float(np.std(raw_rates, ddof=1)),
        "negative_fraction": float(np.mean(raw_rates <= 0.0)),
        "quantiles": np.quantile(raw_rates, [0.05, 0.5, 0.95]),
    }


def main():
    print("Quantum erasure channel: coherent information and capacity")
    print("\nQiskit entropy calculation")
    print(
        "epsilon   S(receiver)   S(reference,receiver)   "
        "I(R>B)       capacity"
    )
    print("-" * 78)

    maximum_error = 0.0
    for epsilon in np.linspace(0.0, 1.0, 11):
        result = coherent_information_qiskit(epsilon)
        analytic = 1.0 - 2.0 * epsilon
        maximum_error = max(
            maximum_error,
            abs(result["coherent_information"] - analytic),
        )
        print(
            f" {epsilon:5.2f}      "
            f"{result['receiver_entropy']:.9f}          "
            f"{result['joint_entropy']:.9f}       "
            f"{result['coherent_information']:+.9f}   "
            f"{quantum_capacity(epsilon):.9f}"
        )

    print(f"\nMaximum error versus 1-2 epsilon: {maximum_error:.3e}")

    epsilon = 0.35
    detailed = coherent_information_qiskit(epsilon)
    expected_receiver_entropy = (
        binary_entropy(epsilon) + 1.0 - epsilon
    )
    expected_joint_entropy = (
        binary_entropy(epsilon) + epsilon
    )

    print(f"\nDetailed spectrum at epsilon={epsilon:.2f}")
    print("Receiver eigenvalues:")
    print(np.round(detailed["receiver_eigenvalues"], 9))
    print("Joint eigenvalues:")
    print(np.round(detailed["joint_eigenvalues"], 9))
    print(f"Trace:                       {detailed['trace']:.12f}")
    print(f"Qiskit S(receiver):          {detailed['receiver_entropy']:.12f}")
    print(f"Analytic h2(e)+1-e:          {expected_receiver_entropy:.12f}")
    print(f"Qiskit S(joint):             {detailed['joint_entropy']:.12f}")
    print(f"Analytic h2(e)+e:            {expected_joint_entropy:.12f}")
    print(
        "Quantum capacity:             "
        f"{quantum_capacity(epsilon):.12f} qubits/use"
    )

    block_size = 1_000
    trials = 5_000
    experiment = finite_block_experiment(
        epsilon,
        block_size,
        trials,
        seed=76,
    )
    print(
        f"\nFinite blocks: {trials:,} trials x "
        f"{block_size:,} channel uses"
    )
    print(f"Mean raw coherent rate:      {experiment['mean']:.9f}")
    print(
        "Rate standard deviation:      "
        f"{experiment['standard_deviation']:.9f}"
    )
    print(
        "Central 90% interval:         "
        f"[{experiment['quantiles'][0]:.9f}, "
        f"{experiment['quantiles'][2]:.9f}]"
    )
    print(
        "Nonpositive block fraction:   "
        f"{experiment['negative_fraction']:.3%}"
    )


if __name__ == "__main__":
    main()
