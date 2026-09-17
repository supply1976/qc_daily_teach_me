"""Day 31: Hadamard-test autocorrelation 與 Hamiltonian spectral function。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector


def hamiltonian_matrix(z_field: float, x_field: float) -> np.ndarray:
    identity = np.eye(2, dtype=complex)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # identity 只用來明確標示資料型別與矩陣維度。
    return 0.0 * identity + z_field * pauli_z + x_field * pauli_x


def evolution_matrix(
    time: float,
    z_field: float,
    x_field: float,
) -> np.ndarray:
    """利用 H^2 = omega^2 I 計算 exp(-i H t)。"""
    hamiltonian = hamiltonian_matrix(z_field, x_field)
    omega = np.sqrt(z_field**2 + x_field**2)
    identity = np.eye(2, dtype=complex)

    return (
        np.cos(omega * time) * identity
        - 1j
        * np.sin(omega * time)
        * hamiltonian
        / omega
    )


def exact_correlation(
    time: float,
    z_field: float,
    x_field: float,
) -> complex:
    """初態為 |0> 時的 <0|exp(-iHt)|0>。"""
    unitary = evolution_matrix(time, z_field, x_field)
    return complex(unitary[0, 0])


def hadamard_test_circuit(
    time: float,
    z_field: float,
    x_field: float,
    component: str,
) -> QuantumCircuit:
    """q0 是 ancilla，q1 是 system；system 初態為 |0>。"""
    if component not in {"real", "imag"}:
        raise ValueError("component must be 'real' or 'imag'")

    unitary = UnitaryGate(
        evolution_matrix(time, z_field, x_field),
        label="U(t)",
    )
    controlled_unitary = unitary.control(1)

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.append(controlled_unitary, [0, 1])

    if component == "imag":
        circuit.sdg(0)
    circuit.h(0)

    return circuit


def sampled_component(
    time: float,
    z_field: float,
    x_field: float,
    component: str,
    shots: int,
    seed: int,
) -> float:
    state = Statevector.from_instruction(
        hadamard_test_circuit(
            time,
            z_field,
            x_field,
            component,
        )
    )
    state.seed(seed)
    counts = state.sample_counts(shots=shots, qargs=[0])

    return float(
        (counts.get("0", 0) - counts.get("1", 0))
        / shots
    )


def sampled_correlation(
    time: float,
    z_field: float,
    x_field: float,
    shots: int,
    seed: int,
) -> complex:
    real = sampled_component(
        time,
        z_field,
        x_field,
        "real",
        shots,
        seed,
    )
    imaginary = sampled_component(
        time,
        z_field,
        x_field,
        "imag",
        shots,
        seed + 1,
    )
    return real + 1j * imaginary


def spectral_transform(
    correlations: np.ndarray,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """計算 |integral dt exp(+i E t) C(t)| 的離散版本。"""
    sample_count = correlations.size
    window = np.hanning(sample_count)

    # times 以零為中心，先移回 FFT 的標準排列。
    ordered = np.fft.ifftshift(window * correlations)
    spectrum = np.abs(
        np.fft.fftshift(np.fft.ifft(ordered))
    )
    energies = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(sample_count, d=time_step)
    )

    return energies, spectrum


def peak_summary(
    energies: np.ndarray,
    spectrum: np.ndarray,
) -> tuple[float, float, float, float]:
    negative = np.flatnonzero(energies < 0)
    positive = np.flatnonzero(energies > 0)

    negative_index = negative[np.argmax(spectrum[negative])]
    positive_index = positive[np.argmax(spectrum[positive])]

    negative_height = float(spectrum[negative_index])
    positive_height = float(spectrum[positive_index])
    total_height = negative_height + positive_height

    return (
        float(energies[negative_index]),
        float(energies[positive_index]),
        negative_height / total_height,
        positive_height / total_height,
    )


def main() -> None:
    z_field = 0.7
    x_field = 1.1
    omega = np.sqrt(z_field**2 + x_field**2)

    # H 的 eigenvalues 是 -omega 與 +omega。
    exact_negative_weight = 0.5 * (1.0 - z_field / omega)
    exact_positive_weight = 0.5 * (1.0 + z_field / omega)

    sample_count = 256
    time_step = 0.15
    shots = 4096
    times = (
        np.arange(sample_count) - sample_count // 2
    ) * time_step

    exact_values = np.array([
        exact_correlation(time, z_field, x_field)
        for time in times
    ])
    sampled_values = np.array([
        sampled_correlation(
            time,
            z_field,
            x_field,
            shots,
            seed=31000 + 2 * index,
        )
        for index, time in enumerate(times)
    ])

    exact_energies, exact_spectrum = spectral_transform(
        exact_values,
        time_step,
    )
    sampled_energies, sampled_spectrum = spectral_transform(
        sampled_values,
        time_step,
    )

    exact_peaks = peak_summary(exact_energies, exact_spectrum)
    sampled_peaks = peak_summary(
        sampled_energies,
        sampled_spectrum,
    )

    maximum_correlation_error = float(
        np.max(np.abs(sampled_values - exact_values))
    )
    rms_correlation_error = float(
        np.sqrt(np.mean(np.abs(sampled_values - exact_values) ** 2))
    )
    energy_resolution = 2.0 * np.pi / (
        sample_count * time_step
    )

    print("Hamiltonian H = 0.7 Z + 1.1 X")
    print(f"Exact energies:       {-omega:+.9f}, {omega:+.9f}")
    print(
        "Exact spectral weights: "
        f"{exact_negative_weight:.9f}, "
        f"{exact_positive_weight:.9f}"
    )
    print(f"Time samples:         {sample_count}")
    print(f"Time step:            {time_step:.3f}")
    print(f"Energy resolution:    {energy_resolution:.9f}")
    print(f"Shots per component:  {shots}")

    print("\nCorrelation checkpoints:")
    for time in [0.0, 0.9, 1.8]:
        index = int(round(time / time_step)) + sample_count // 2
        exact = exact_values[index]
        sampled = sampled_values[index]
        print(
            f"t={time:3.1f}: "
            f"exact={exact.real:+.6f}{exact.imag:+.6f}j, "
            f"sampled={sampled.real:+.6f}{sampled.imag:+.6f}j"
        )

    print("\nHadamard-test correlation errors:")
    print(f"Maximum error: {maximum_correlation_error:.6f}")
    print(f"RMS error:     {rms_correlation_error:.6f}")

    print("\nFourier spectral peaks:")
    print("                 negative E               positive E")
    print(
        "Exact data:   "
        f"E={exact_peaks[0]:+.9f}, w={exact_peaks[2]:.9f}    "
        f"E={exact_peaks[1]:+.9f}, w={exact_peaks[3]:.9f}"
    )
    print(
        "Sampled data: "
        f"E={sampled_peaks[0]:+.9f}, w={sampled_peaks[2]:.9f}    "
        f"E={sampled_peaks[1]:+.9f}, w={sampled_peaks[3]:.9f}"
    )


if __name__ == "__main__":
    main()
