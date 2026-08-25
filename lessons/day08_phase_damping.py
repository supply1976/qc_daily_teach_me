"""Day 08：量子雜訊、Kraus operators 與 phase damping。"""

import numpy as np
from qiskit.quantum_info import DensityMatrix, Kraus, Pauli, Statevector


def phase_damping_channel(lam: float) -> Kraus:
    """建立 phase-damping channel；lam 必須位於 [0, 1]。"""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be between 0 and 1")

    k0 = np.array(
        [[1.0, 0.0], [0.0, np.sqrt(1.0 - lam)]],
        dtype=complex,
    )
    k1 = np.array(
        [[0.0, 0.0], [0.0, np.sqrt(lam)]],
        dtype=complex,
    )
    return Kraus([k0, k1])


def main() -> None:
    # |+> = (|0> + |1>) / sqrt(2)，具有最大的 X-basis coherence。
    initial_state = DensityMatrix(Statevector.from_label("+"))
    pauli_x = Pauli("X")

    print("Initial |+><+| density matrix:")
    print(np.round(initial_state.data, 6))

    print("\n lam    P(0)    P(1)   |rho01|   purity     <X>")
    print("--------------------------------------------------")

    for lam in [0.0, 0.25, 0.50, 0.75, 1.0]:
        channel = phase_damping_channel(lam)
        noisy_state = initial_state.evolve(channel)

        probabilities = noisy_state.probabilities()
        coherence = abs(noisy_state.data[0, 1])
        purity = np.real_if_close(noisy_state.purity()).item()
        expectation_x = np.real_if_close(
            noisy_state.expectation_value(pauli_x)
        ).item()

        print(
            f"{lam:5.2f}  {probabilities[0]:6.3f}  {probabilities[1]:6.3f}"
            f"   {coherence:7.4f}   {purity:7.4f}  {expectation_x:7.4f}"
        )

    fully_dephased = initial_state.evolve(phase_damping_channel(1.0))
    print("\nFully dephased density matrix (lam=1):")
    print(np.round(fully_dephased.data, 6))


if __name__ == "__main__":
    main()
