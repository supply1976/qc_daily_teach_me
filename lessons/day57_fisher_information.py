"""Day 57: Fisher information and the Cramer-Rao bound."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector


def ramsey_circuit(error_angle, sequence_length, control_phase):
    """Ramsey circuit probing L*error_angle + control_phase."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    for _ in range(sequence_length):
        circuit.rz(float(error_angle), 0)
    circuit.rz(float(control_phase), 0)
    circuit.h(0)
    return circuit


def qiskit_probability_zero(
    error_angle,
    sequence_length,
    control_phase,
):
    state = Statevector.from_instruction(
        ramsey_circuit(
            error_angle,
            sequence_length,
            control_phase,
        )
    )
    return float(state.probabilities()[0])


def probability_zero(
    error_angle,
    sequence_length,
    control_phase,
    depolarizing_alpha,
):
    contrast = depolarizing_alpha**sequence_length
    phase = sequence_length * error_angle + control_phase
    return 0.5 * (1.0 + contrast * np.cos(phase))


def classical_fisher_information(
    error_angle,
    sequence_length,
    control_phase,
    depolarizing_alpha,
):
    """Fisher information of one binary Ramsey measurement."""
    contrast = depolarizing_alpha**sequence_length
    phase = sequence_length * error_angle + control_phase
    probability = probability_zero(
        error_angle,
        sequence_length,
        control_phase,
        depolarizing_alpha,
    )
    derivative = (
        -0.5
        * contrast
        * sequence_length
        * np.sin(phase)
    )
    variance = probability * (1.0 - probability)
    if variance < 1e-15:
        return 0.0
    return float(derivative**2 / variance)


def dephased_phase_state(
    error_angle,
    sequence_length,
    depolarizing_alpha,
):
    """Density matrix before the final Ramsey analysis gates."""
    contrast = depolarizing_alpha**sequence_length
    phase = sequence_length * error_angle
    coherence = contrast * np.exp(-1j * phase)
    return DensityMatrix(
        0.5
        * np.array(
            [
                [1.0, coherence],
                [np.conj(coherence), 1.0],
            ],
            dtype=complex,
        )
    )


def density_derivative(
    error_angle,
    sequence_length,
    depolarizing_alpha,
):
    contrast = depolarizing_alpha**sequence_length
    phase = sequence_length * error_angle
    upper = (
        -0.5j
        * sequence_length
        * contrast
        * np.exp(-1j * phase)
    )
    return np.array(
        [
            [0.0, upper],
            [np.conj(upper), 0.0],
        ],
        dtype=complex,
    )


def quantum_fisher_information(
    error_angle,
    sequence_length,
    depolarizing_alpha,
):
    """Evaluate QFI with its eigenbasis spectral formula."""
    density = dephased_phase_state(
        error_angle,
        sequence_length,
        depolarizing_alpha,
    ).data
    derivative = density_derivative(
        error_angle,
        sequence_length,
        depolarizing_alpha,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(density)

    information = 0.0
    for left in range(2):
        for right in range(2):
            denominator = eigenvalues[left] + eigenvalues[right]
            if denominator > 1e-14:
                matrix_element = np.vdot(
                    eigenvectors[:, left],
                    derivative @ eigenvectors[:, right],
                )
                information += (
                    2.0
                    * abs(matrix_element) ** 2
                    / denominator
                )
    return float(np.real(information))


def local_monte_carlo(
    nominal_error,
    sequence_length,
    depolarizing_alpha,
    shots,
    repetitions,
    rng,
):
    """Local estimator at the quadrature point of the Ramsey fringe."""
    contrast = depolarizing_alpha**sequence_length
    control_phase = (
        np.pi / 2.0 - sequence_length * nominal_error
    )
    probability = float(
        probability_zero(
            nominal_error,
            sequence_length,
            control_phase,
            depolarizing_alpha,
        )
    )
    zero_counts = rng.binomial(
        shots,
        probability,
        size=repetitions,
    )
    measured_probability = zero_counts / shots

    # Linearize p0 around the quadrature point, where
    # dp0/depsilon = -C_L*L/2.
    estimates = nominal_error - (
        2.0
        * (measured_probability - 0.5)
        / (contrast * sequence_length)
    )
    return estimates


def main():
    true_error = 0.073
    depolarizing_alpha = 0.98
    maximum_length = 128
    gate_budget = 102_400
    repetitions = 4_000
    seed = 20261013

    lengths = np.arange(1, maximum_length + 1)
    information_per_shot = (
        lengths**2
        * depolarizing_alpha ** (2 * lengths)
    )
    information_per_gate = information_per_shot / lengths
    best_fixed_shots = int(
        lengths[np.argmax(information_per_shot)]
    )
    best_fixed_gates = int(
        lengths[np.argmax(information_per_gate)]
    )

    continuous_fixed_shots = -1.0 / np.log(
        depolarizing_alpha
    )
    continuous_fixed_gates = -1.0 / (
        2.0 * np.log(depolarizing_alpha)
    )

    # Validate ideal Ramsey probabilities with actual Qiskit circuits.
    qiskit_errors = []
    for length in [1, 8, 25, 50, 100]:
        phase = np.pi / 2.0 - length * true_error
        qiskit = qiskit_probability_zero(
            true_error,
            length,
            phase,
        )
        analytic = probability_zero(
            true_error,
            length,
            phase,
            depolarizing_alpha=1.0,
        )
        qiskit_errors.append(abs(qiskit - analytic))

    # At quadrature the Ramsey Z measurement reaches the QFI bound.
    qfi_errors = []
    saturation_errors = []
    for length in [1, 8, 25, 50, 100]:
        control_phase = np.pi / 2.0 - length * true_error
        classical = classical_fisher_information(
            true_error,
            length,
            control_phase,
            depolarizing_alpha,
        )
        quantum = quantum_fisher_information(
            true_error,
            length,
            depolarizing_alpha,
        )
        analytic = (
            length**2
            * depolarizing_alpha ** (2 * length)
        )
        qfi_errors.append(abs(quantum - analytic))
        saturation_errors.append(abs(classical - quantum))

    rng = np.random.default_rng(seed)
    comparison_lengths = [1, best_fixed_gates, best_fixed_shots, 100]
    results = []
    for length in comparison_lengths:
        shots = gate_budget // length
        information = (
            length**2
            * depolarizing_alpha ** (2 * length)
        )
        total_information = shots * information
        cramer_rao_std = 1.0 / np.sqrt(total_information)
        estimates = local_monte_carlo(
            true_error,
            length,
            depolarizing_alpha,
            shots,
            repetitions,
            rng,
        )
        results.append({
            "length": length,
            "contrast": depolarizing_alpha**length,
            "shots": shots,
            "information": information,
            "total_information": total_information,
            "bound": cramer_rao_std,
            "observed": float(np.std(estimates, ddof=1)),
            "bias": float(np.mean(estimates) - true_error),
        })

    print("Fisher information for noisy Ramsey calibration")
    print(f"True angle error:          {true_error:+.9f} rad/gate")
    print(f"Per-gate Bloch shrink:     {depolarizing_alpha:.6f}")
    print(f"Fixed gate-use budget:     {gate_budget}")
    print(f"Monte Carlo repetitions:   {repetitions}")
    print(
        "Maximum Qiskit-versus-analytic probability error: "
        f"{max(qiskit_errors):.3e}"
    )
    print(
        "Maximum spectral-QFI formula error:               "
        f"{max(qfi_errors):.3e}"
    )
    print(
        "Maximum |classical FI - quantum FI| at quadrature: "
        f"{max(saturation_errors):.3e}"
    )

    print("\nOptimal sequence length:")
    print(
        " fixed shots: "
        f"continuous={continuous_fixed_shots:.6f}, "
        f"integer={best_fixed_shots}"
    )
    print(
        " fixed gate uses: "
        f"continuous={continuous_fixed_gates:.6f}, "
        f"integer={best_fixed_gates}"
    )

    print("\nMonte Carlo with the same total gate-use budget:")
    print(
        "   L   contrast    shots     FI/shot"
        "      total FI     CRLB std   observed std      bias"
    )
    print("-" * 96)
    for result in results:
        print(
            f"{result['length']:4d}"
            f"   {result['contrast']:.6f}"
            f"   {result['shots']:6d}"
            f"   {result['information']:10.3f}"
            f"   {result['total_information']:11.3f}"
            f"   {result['bound']:.3e}"
            f"      {result['observed']:.3e}"
            f"   {result['bias']:+.2e}"
        )

    print(
        "\nLonger circuits can carry more information per shot, "
        "while shorter circuits can win per gate application."
    )


if __name__ == "__main__":
    main()
