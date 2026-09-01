"""
Benchmark: Statevector Simulation Cost vs. Number of Qubits
===============================================================
A statevector simulation has to store one complex amplitude per basis
state, and an n-qubit system has 2^n basis states. That means:

    10 qubits ->     1,024 amplitudes
    20 qubits -> 1,048,576 amplitudes
    30 qubits -> 1,073,741,824 amplitudes

So the memory *and* time cost of exact statevector simulation grows
exponentially with the number of qubits - this is the same reason full
classical simulation of quantum computers can't scale indefinitely, and
it's part of why quantum computers are interesting in the first place.

This script builds circuits of increasing width, times how long it
takes Qiskit to compute the exact statevector for each one, and plots
execution time (log scale) against qubit count, so the exponential
trend is visible directly.

Methodology notes:
- Circuit *construction* is done before starting the timer, so we're
  only timing the statevector computation itself, not the (comparatively
  cheap) work of building the circuit.
- Each width is timed several times and the median is used, which is
  more robust to one-off system noise (background processes, garbage
  collection, etc.) than a single measurement.
"""

import statistics
import time

from quantum_circuit_simulator import CircuitSimulator
from quantum_circuit_simulator.visualization import save_benchmark_plot

REPEATS = 5


def build_circuit(num_qubits: int) -> CircuitSimulator:
    """
    Builds a circuit with a modest, fixed amount of entangling
    structure (not just an empty register), so the benchmark reflects
    a somewhat realistic circuit rather than the trivial all-zero state.
    """
    circuit = CircuitSimulator(num_qubits, name=f"benchmark_{num_qubits}q")
    circuit.h(0)
    for q in range(num_qubits - 1):
        circuit.cx(q, q + 1)
    return circuit


def time_statevector(circuit: CircuitSimulator, repeats: int = REPEATS) -> float:
    """
    Times only `get_statevector()` (circuit construction happens before
    this function is called), and returns the median of `repeats` runs.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        circuit.get_statevector()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def run_benchmark(qubit_counts=None):
    if qubit_counts is None:
        qubit_counts = [2, 4, 6, 8, 10, 12, 14, 16, 18]

    print(f"{'Qubits':<10}{'Amplitudes (2^n)':<20}{'Median time (s)':<18}")

    results = []
    for n in qubit_counts:
        circuit = build_circuit(n)  # construction NOT timed
        median_time = time_statevector(circuit)
        results.append((n, median_time))
        print(f"{n:<10}{2 ** n:<20}{median_time:<18.6f}")

    qubit_values = [n for n, _ in results]
    time_values = [t for _, t in results]
    filename = save_benchmark_plot(qubit_values, time_values, "benchmark_qubits_vs_time.png")
    print(f"\n[Saved benchmark plot -> {filename}]")
    print(
        "\nExecution time should roughly follow a straight line on this "
        "log-scale plot, which is the visual signature of exponential "
        "growth (2^n amplitudes) - not a fixed per-qubit cost, but a "
        "cost that doubles with every additional qubit."
    )
    return results


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
