"""Day 30: Hadamard test 量測複數 unitary expectation value。"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import Operator, Statevector


def system_state(theta: float) -> Statevector:
    """準備 |psi> = Ry(theta)|0>。"""
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    return Statevector.from_instruction(circuit)


def exact_expectation(theta: float, phase: float) -> complex:
    """用 statevector 計算 <psi|Rz(phase)|psi>，只作驗證。"""
    state = system_state(theta)
    unitary = Operator(RZGate(phase)).data
    return complex(np.vdot(state.data, unitary @ state.data))


def hadamard_test_circuit(
    theta: float,
    phase: float,
    component: str,
) -> QuantumCircuit:
    """Ancilla q0；system q1。component 為 real 或 imag。"""
    if component not in {"real", "imag"}:
        raise ValueError("component must be 'real' or 'imag'")

    circuit = QuantumCircuit(2)

    # 準備 system state |psi>。
    circuit.ry(theta, 1)

    # (|0>|psi> + |1>U|psi>) / sqrt(2)
    circuit.h(0)
    circuit.crz(phase, 0, 1)

    # X measurement 給 Re；Y measurement 給 Im。
    if component == "imag":
        circuit.sdg(0)
    circuit.h(0)

    return circuit


def exact_ancilla_expectation(
    theta: float,
    phase: float,
    component: str,
) -> float:
    """由完整電路的 ancilla probabilities 計算 P(0)-P(1)。"""
    state = Statevector.from_instruction(
        hadamard_test_circuit(theta, phase, component)
    )
    probabilities = state.probabilities(qargs=[0])
    return float(probabilities[0] - probabilities[1])


def sampled_ancilla_expectation(
    theta: float,
    phase: float,
    component: str,
    shots: int,
    seed: int,
) -> tuple[float, dict[str, int]]:
    """以 computational-basis samples 估計 ancilla 的 Z expectation。"""
    state = Statevector.from_instruction(
        hadamard_test_circuit(theta, phase, component)
    )
    state.seed(seed)
    raw_counts = state.sample_counts(shots=shots, qargs=[0])
    counts = {
        str(outcome): int(count)
        for outcome, count in raw_counts.items()
    }

    estimate = (
        counts.get("0", 0) - counts.get("1", 0)
    ) / shots

    return estimate, counts


def main() -> None:
    theta = 1.1
    phase = 1.3

    exact = exact_expectation(theta, phase)
    real_from_circuit = exact_ancilla_expectation(
        theta,
        phase,
        "real",
    )
    imaginary_from_circuit = exact_ancilla_expectation(
        theta,
        phase,
        "imag",
    )

    analytic = (
        np.cos(phase / 2.0)
        - 1j
        * np.cos(theta)
        * np.sin(phase / 2.0)
    )
    maximum_exact_error = max(
        abs(exact - analytic),
        abs(exact.real - real_from_circuit),
        abs(exact.imag - imaginary_from_circuit),
    )

    print(f"State angle theta: {theta:.6f}")
    print(f"Rz phase phi:      {phase:.6f}")
    print(f"Exact <psi|U|psi>: {exact.real:+.9f}{exact.imag:+.9f}j")
    print(
        "Analytic value:     "
        f"{analytic.real:+.9f}{analytic.imag:+.9f}j"
    )
    print(
        "Ancilla exact:      "
        f"{real_from_circuit:+.9f}"
        f"{imaginary_from_circuit:+.9f}j"
    )
    print(
        "Maximum exact error: "
        f"{maximum_exact_error:.3e}"
    )

    print("\nFinite-shot Hadamard tests:")
    print(" shots      Re estimate      Im estimate      complex error")
    print("-------------------------------------------------------------")

    for index, shots in enumerate([256, 1024, 4096, 16384]):
        real_estimate, _ = sampled_ancilla_expectation(
            theta,
            phase,
            "real",
            shots,
            seed=3000 + 2 * index,
        )
        imaginary_estimate, _ = sampled_ancilla_expectation(
            theta,
            phase,
            "imag",
            shots,
            seed=3001 + 2 * index,
        )
        estimate = real_estimate + 1j * imaginary_estimate

        print(
            f"{shots:6d}   "
            f"{real_estimate:+.9f}   "
            f"{imaginary_estimate:+.9f}   "
            f"{abs(estimate - exact):.3e}"
        )

    # Rz(phi) = cos(phi/2) I - i sin(phi/2) Z，
    # 因此 imaginary part 可反推出 <Z>。
    shots = 16384
    imaginary_estimate, counts = sampled_ancilla_expectation(
        theta,
        phase,
        "imag",
        shots,
        seed=3099,
    )
    reconstructed_z = (
        -imaginary_estimate / np.sin(phase / 2.0)
    )

    print("\nObservable reconstruction:")
    print(f"Imaginary counts: {counts}")
    print(f"Exact <Z>:        {np.cos(theta):+.9f}")
    print(f"Reconstructed <Z>: {reconstructed_z:+.9f}")


if __name__ == "__main__":
    main()
