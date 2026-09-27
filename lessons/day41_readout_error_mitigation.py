"""Day 41: readout-error mitigation for shot-based QBM observables."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp


def hamiltonian(coupling=0.9, transverse_field=0.65):
    return SparsePauliOp.from_list([
        ("ZZ", -float(coupling)),
        ("IX", -float(transverse_field)),
        ("XI", -float(transverse_field)),
    ]).to_matrix()


def gibbs_state(beta=1.0):
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian())
    weights = np.exp(-beta * (eigenvalues - eigenvalues[0]))
    density = (
        eigenvectors
        @ np.diag(weights / np.sum(weights))
        @ eigenvectors.conj().T
    )
    return DensityMatrix(density)


def expectation(density, label):
    operator = SparsePauliOp.from_list([(label, 1.0)])
    return float(np.real(density.expectation_value(operator)))


def basis_probabilities(density, basis):
    if basis == "Z":
        rotated = density
    elif basis == "X":
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])
        rotated = density.evolve(circuit)
    else:
        raise ValueError("basis must be 'Z' or 'X'")
    probabilities = np.real(np.diag(rotated.data))
    return probabilities / np.sum(probabilities)


def local_readout_matrix(error_probability):
    """M[y, x] = probability of observing y when the true bit is x."""
    error = float(error_probability)
    return np.array([
        [1.0 - error, error],
        [error, 1.0 - error],
    ])


def two_qubit_readout_matrix(error_probability):
    single = local_readout_matrix(error_probability)
    return np.kron(single, single)


def noisy_frequencies(probabilities, readout_matrix, shots, seed):
    observed_probabilities = readout_matrix @ probabilities
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(shots, observed_probabilities)
    return counts / shots


def project_probability_simplex(values):
    """Euclidean projection onto p_i >= 0 and sum_i p_i = 1."""
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values)
    indices = np.arange(1, values.size + 1)
    keep = sorted_values - (cumulative - 1.0) / indices > 0
    rho = np.nonzero(keep)[0][-1]
    threshold = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(values - threshold, 0.0)


def mitigate(frequencies, calibration_matrix):
    quasi_probabilities = np.linalg.pinv(calibration_matrix) @ frequencies
    return project_probability_simplex(quasi_probabilities)


def observable_from_probabilities(probabilities, basis):
    outcomes = np.arange(4)
    q0 = outcomes & 1
    q1 = (outcomes >> 1) & 1
    if basis == "Z":
        values = (1 - 2 * q0) * (1 - 2 * q1)
    else:
        # Report the average (X0 + X1) / 2.
        values = ((1 - 2 * q0) + (1 - 2 * q1)) / 2
    return float(probabilities @ values)


def calibrate_readout(true_matrix, shots, seed):
    """Estimate each column using prepared computational basis states."""
    rng = np.random.default_rng(seed)
    estimated = np.zeros_like(true_matrix)
    for prepared_state in range(4):
        counts = rng.multinomial(
            shots,
            true_matrix[:, prepared_state],
        )
        estimated[:, prepared_state] = counts / shots
    return estimated


def repeated_experiment(
    probabilities,
    basis,
    true_readout,
    calibration,
    shots,
    repeats,
    seed,
):
    raw_values = []
    mitigated_values = []
    for repeat in range(repeats):
        frequencies = noisy_frequencies(
            probabilities,
            true_readout,
            shots,
            seed + repeat,
        )
        raw_values.append(
            observable_from_probabilities(frequencies, basis)
        )
        mitigated_values.append(
            observable_from_probabilities(
                mitigate(frequencies, calibration),
                basis,
            )
        )
    return np.asarray(raw_values), np.asarray(mitigated_values)


def main():
    density = gibbs_state()
    readout_error = 0.08
    shots = 2048
    repeats = 500

    true_readout = two_qubit_readout_matrix(readout_error)
    calibration = calibrate_readout(
        true_readout,
        shots=50_000,
        seed=4100,
    )

    print(f"Readout bit-flip probability: {readout_error:.3f}")
    print("Exact two-qubit readout matrix:")
    print(np.array2string(true_readout, precision=5))
    print("Maximum calibration error:")
    print(f"{np.max(np.abs(calibration - true_readout)):.3e}\n")

    print("Observable estimates over 500 repeated experiments:")
    print(" observable      exact       raw mean     mitigated mean   raw std    mitigated std")
    print("-" * 91)

    results = {}
    for basis, label, exact in [
        ("Z", "<ZZ>", expectation(density, "ZZ")),
        (
            "X",
            "avg <X>",
            0.5 * (
                expectation(density, "IX")
                + expectation(density, "XI")
            ),
        ),
    ]:
        probabilities = basis_probabilities(density, basis)
        raw, corrected = repeated_experiment(
            probabilities,
            basis,
            true_readout,
            calibration,
            shots,
            repeats,
            seed=4200 if basis == "Z" else 5200,
        )
        results[basis] = (exact, raw, corrected)
        print(
            f" {label:9s}   {exact:+.6f}   {np.mean(raw):+.6f}"
            f"      {np.mean(corrected):+.6f}      {np.std(raw, ddof=1):.6f}"
            f"     {np.std(corrected, ddof=1):.6f}"
        )

    attenuation = 1.0 - 2.0 * readout_error
    print("\nAnalytic attenuation factors:")
    print(f"  one-qubit Pauli: (1-2e)   = {attenuation:.6f}")
    print(f"  two-qubit Pauli: (1-2e)^2 = {attenuation**2:.6f}")

    exact_gradient = np.zeros(2)
    raw_gradient = np.array([
        np.mean(results["Z"][1]) - results["Z"][0],
        2.0 * (np.mean(results["X"][1]) - results["X"][0]),
    ])
    mitigated_gradient = np.array([
        np.mean(results["Z"][2]) - results["Z"][0],
        2.0 * (np.mean(results["X"][2]) - results["X"][0]),
    ])

    print("\nEstimated QBM gradient at the true parameters:")
    print("  ideal:     ", np.array2string(exact_gradient, precision=6))
    print("  raw:       ", np.array2string(raw_gradient, precision=6))
    print("  mitigated: ", np.array2string(mitigated_gradient, precision=6))


if __name__ == "__main__":
    main()
