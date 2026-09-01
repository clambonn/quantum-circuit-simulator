"""
Tests for the validation logic: num_qubits, qubit indices, control/target
distinctness, rotation angles, and shots.
"""

import math

import pytest

from quantum_circuit_simulator import CircuitSimulator

# ---------- num_qubits validation ----------

def test_num_qubits_zero_raises():
    with pytest.raises(ValueError):
        CircuitSimulator(0)


def test_num_qubits_negative_raises():
    with pytest.raises(ValueError):
        CircuitSimulator(-2)


def test_num_qubits_non_int_raises():
    with pytest.raises(TypeError):
        CircuitSimulator(2.5)


def test_num_qubits_bool_raises():
    """bool is technically an int subclass in Python, but True/False as
    a qubit count makes no physical sense and should be rejected."""
    with pytest.raises(TypeError):
        CircuitSimulator(True)


def test_num_qubits_valid_ok():
    c = CircuitSimulator(3)
    assert c.num_qubits == 3


# ---------- qubit index validation ----------

def test_qubit_out_of_range_raises():
    c = CircuitSimulator(3)
    with pytest.raises(ValueError):
        c.h(10)


def test_qubit_negative_raises():
    c = CircuitSimulator(3)
    with pytest.raises(ValueError):
        c.h(-1)


def test_qubit_non_int_raises():
    c = CircuitSimulator(3)
    with pytest.raises(TypeError):
        c.h(1.5)


def test_qubit_bool_raises():
    c = CircuitSimulator(3)
    with pytest.raises(TypeError):
        c.h(True)


def test_qubit_valid_ok():
    c = CircuitSimulator(3)
    c.h(0)  # should not raise
    assert "H(q0)" in c.history


# ---------- control/target distinctness ----------

def test_cx_same_qubit_raises():
    c = CircuitSimulator(3)
    with pytest.raises(ValueError):
        c.cx(0, 0)


def test_cz_same_qubit_raises():
    c = CircuitSimulator(3)
    with pytest.raises(ValueError):
        c.cz(1, 1)


def test_swap_same_qubit_raises():
    c = CircuitSimulator(3)
    with pytest.raises(ValueError):
        c.swap(0, 0)


def test_toffoli_repeated_qubits_raises():
    c = CircuitSimulator(3)
    with pytest.raises(ValueError):
        c.toffoli(0, 0, 1)


def test_toffoli_distinct_qubits_ok():
    c = CircuitSimulator(3)
    c.toffoli(0, 1, 2)  # should not raise


# ---------- rotation angle validation (Rx, Ry, Rz) ----------

def test_rotation_angle_must_be_numeric():
    c = CircuitSimulator(1)
    with pytest.raises(TypeError):
        c.rx(0, "invalid")


def test_rotation_angle_bool_raises():
    c = CircuitSimulator(1)
    with pytest.raises(TypeError):
        c.ry(0, True)


def test_rotation_angle_must_be_finite_nan():
    c = CircuitSimulator(1)
    with pytest.raises(ValueError):
        c.rx(0, float("nan"))


def test_rotation_angle_must_be_finite_infinity():
    c = CircuitSimulator(1)
    with pytest.raises(ValueError):
        c.rz(0, float("inf"))


def test_rotation_angle_valid_ok():
    c = CircuitSimulator(1)
    c.rx(0, math.pi / 2)  # should not raise
    assert any("Rx" in g for g in c.history)


# ---------- shots validation (run_measurement) ----------

def test_shots_zero_raises():
    c = CircuitSimulator(2)
    c.h(0)
    with pytest.raises(ValueError):
        c.run_measurement(shots=0)


def test_shots_negative_raises():
    c = CircuitSimulator(2)
    c.h(0)
    with pytest.raises(ValueError):
        c.run_measurement(shots=-10)


def test_shots_non_int_raises():
    c = CircuitSimulator(2)
    c.h(0)
    with pytest.raises(TypeError):
        c.run_measurement(shots="100")


def test_shots_valid_ok():
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    counts = c.run_measurement(shots=100)
    assert sum(counts.values()) == 100


# ---------- shots validation (compare_theoretical_vs_measured) ----------

def test_compare_shots_zero_raises():
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    with pytest.raises(ValueError):
        c.compare_theoretical_vs_measured(shots=0)


# ---------- qubits parameter validation (partial measurement) ----------

def test_qubits_empty_list_raises():
    c = CircuitSimulator(2)
    c.h(0)
    with pytest.raises(ValueError):
        c.run_measurement(shots=100, qubits=[])


def test_qubits_non_list_raises():
    c = CircuitSimulator(2)
    c.h(0)
    with pytest.raises(TypeError):
        c.run_measurement(shots=100, qubits=5)


def test_qubits_out_of_range_raises():
    c = CircuitSimulator(2)
    c.h(0)
    with pytest.raises(ValueError):
        c.run_measurement(shots=100, qubits=[5])


def test_qubits_duplicate_raises():
    """Measuring the same qubit twice into two classical bits doesn't
    make physical sense and should be rejected."""
    c = CircuitSimulator(2)
    c.h(0)
    with pytest.raises(ValueError):
        c.run_measurement(shots=100, qubits=[0, 0])


def test_qubits_valid_partial_measurement_ok():
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    counts = c.run_measurement(shots=100, qubits=[0])
    assert sum(counts.values()) == 100
    # partial measurement of 1 qubit -> outcomes are single-character bitstrings
    assert all(len(outcome) == 1 for outcome in counts)
