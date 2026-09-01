"""
Bell State Example
====================
The simplest example of entanglement: H on q0 creates superposition, then
CNOT(q0 -> q1) links the two together. Result: only 50% |00> and 50% |11>
(no |01> or |10> at all) — if we measure one, we immediately know the other.
"""

from quantum_circuit_simulator import CircuitSimulator


def main():
    bell = CircuitSimulator(num_qubits=2, name="bell_state")
    bell.h(0).cx(0, 1)

    bell.summary()
    bell.show_circuit()
    bell.show_statevector()

    print("\nExact probabilities:", bell.get_probabilities())
    bell.show_comparison(shots=4096)


if __name__ == "__main__":
    main()
