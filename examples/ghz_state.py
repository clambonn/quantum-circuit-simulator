"""
GHZ State Example
====================
A generalization of the Bell state idea to 3 qubits instead of two. H on
q0 followed by chained CNOTs (q0->q1, q1->q2) produces a state entangled
across all three: only 50% |000> and 50% |111>.
"""

from quantum_circuit_simulator import CircuitSimulator


def main():
    ghz = CircuitSimulator(num_qubits=3, name="ghz_state")
    ghz.h(0).cx(0, 1).cx(1, 2)

    ghz.summary()
    ghz.show_circuit()
    ghz.show_statevector()

    probabilities = ghz.get_probabilities()
    print("\nExact probabilities:", probabilities)  # expect ~0.5 for |000> and |111>

    ghz.show_comparison(shots=4096)


if __name__ == "__main__":
    main()
