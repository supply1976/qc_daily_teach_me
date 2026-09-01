"""第 15 課：Bernstein-Vazirani、phase kickback 與 hidden string。"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def bernstein_vazirani_circuit(secret: str) -> QuantumCircuit:
    """建立可找出 hidden string 的 Bernstein-Vazirani circuit。"""
    if not secret or any(bit not in "01" for bit in secret):
        raise ValueError("secret must be a non-empty binary string")

    number_of_data_qubits = len(secret)
    ancilla = number_of_data_qubits
    circuit = QuantumCircuit(number_of_data_qubits + 1)

    # 1. 將 ancilla 準備為 |-> = H|1>。
    circuit.x(ancilla)
    circuit.h(ancilla)

    # 2. 將 data register 準備成所有 x 的均勻疊加。
    circuit.h(range(number_of_data_qubits))
    circuit.barrier()

    # 3. Oracle: |x>|y> -> |x>|y xor f_s(x)>。
    # secret 字串寫成 s_(n-1)...s_0，因此要反轉後對應 q0, q1, ...。
    for qubit, bit in enumerate(reversed(secret)):
        if bit == "1":
            circuit.cx(qubit, ancilla)
    circuit.barrier()

    # 4. Hadamard interference 將 phase pattern 解碼成 |s>。
    circuit.h(range(number_of_data_qubits))

    return circuit


def main() -> None:
    secret = "101101"
    shots = 1024
    data_qubits = list(range(len(secret)))

    circuit = bernstein_vazirani_circuit(secret)
    state = Statevector.from_instruction(circuit)

    # 只取樣 data register；ancilla 不包含在輸出中。
    exact_probabilities = {
        str(outcome): float(probability)
        for outcome, probability in state.probabilities_dict(
            qargs=data_qubits,
            decimals=12,
        ).items()
        if probability > 1e-12
    }

    state.seed(1500)
    counts = {
        str(outcome): int(count)
        for outcome, count in state.sample_counts(
            shots=shots,
            qargs=data_qubits,
        ).items()
    }

    recovered_secret = max(counts, key=counts.get)

    print("Bernstein-Vazirani circuit:")
    print(circuit.draw(output="text"))
    print(f"\nHidden string s:       {secret}")
    print(f"Exact probabilities:   {exact_probabilities}")
    print(f"Counts ({shots} shots): {counts}")
    print(f"Recovered string:      {recovered_secret}")
    print(f"Recovery successful:   {recovered_secret == secret}")
    print("\nDeterministic query complexity:")
    print(f"  Classical oracle queries: {len(secret)}")
    print("  Quantum oracle queries:   1")


if __name__ == "__main__":
    main()
