"""
Tests for measurement, theoretical probabilities, and the
theoretical-vs-measured comparison. Statistical tests use a fixed
`seed_simulator` wherever possible so results are reproducible and the
tests aren't flaky.
"""

import pytest

from quantum_circuit_simulator import CircuitSimulator

# ---------- get_probabilities() ----------

def test_theoretical_probabilities_sum_to_one():
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    probabilities = c.get_probabilities()
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_theoretical_probabilities_bell_state():
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    probabilities = c.get_probabilities()
    assert probabilities == pytest.approx({"00": 0.5, "11": 0.5})


# ---------- run_measurement() ----------

def test_measurement_counts_sum_to_shots():
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    counts = c.run_measurement(shots=2048, seed_simulator=42)
    assert sum(counts.values()) == 2048


def test_measurement_only_produces_valid_outcomes():
    """A Bell state should only ever measure 00 or 11, never 01/10."""
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    counts = c.run_measurement(shots=2048, seed_simulator=42)
    assert set(counts.keys()) <= {"00", "11"}


def test_measurement_is_reproducible_with_same_seed():
    """Same circuit, same shots, same seed -> identical counts."""
    c1 = CircuitSimulator(2)
    c1.h(0).cx(0, 1)
    counts1 = c1.run_measurement(shots=1024, seed_simulator=99)

    c2 = CircuitSimulator(2)
    c2.h(0).cx(0, 1)
    counts2 = c2.run_measurement(shots=1024, seed_simulator=99)

    assert counts1 == counts2


def test_measurement_deterministic_circuit():
    """A circuit with no superposition should always give the same
    single outcome, regardless of seed or shots."""
    c = CircuitSimulator(2)
    c.x(0)
    counts = c.run_measurement(shots=500, seed_simulator=1)
    assert counts == {"01": 500}


def test_partial_measurement_only_measures_requested_qubits():
    c = CircuitSimulator(3)
    c.x(0).x(2)  # q0=1, q1=0, q2=1
    counts = c.run_measurement(shots=200, qubits=[0, 1], seed_simulator=5)
    # measuring only q0 (=1) and q1 (=0); Qiskit displays bits in reverse
    # order (c[1]c[0]), so the expected bitstring is "01"
    assert set(counts.keys()) == {"01"}


# ---------- compare_theoretical_vs_measured() ----------

def test_compare_returns_expected_keys():
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    result = c.compare_theoretical_vs_measured(shots=4096, seed_simulator=42)
    for key in (
        "theoretical",
        "measured",
        "total_absolute_error",
        "mean_absolute_error",
        "max_absolute_error",
        "total_variation_distance",
    ):
        assert key in result


def test_compare_tvd_is_small_for_many_shots_with_fixed_seed():
    """With a fixed seed and a reasonably large number of shots, the
    measured distribution should be statistically close to theoretical -
    this is a bound on ordinary sampling noise, not a test of 'accuracy'."""
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    result = c.compare_theoretical_vs_measured(shots=8192, seed_simulator=42)
    assert result["total_variation_distance"] < 0.05


def test_compare_tvd_zero_for_deterministic_circuit():
    """A circuit with a single definite outcome should measure exactly
    the theoretical distribution every time (TVD = 0), independent of seed."""
    c = CircuitSimulator(2)
    c.x(0).x(1)
    result = c.compare_theoretical_vs_measured(shots=1000, seed_simulator=7)
    assert result["total_variation_distance"] == pytest.approx(0.0)
