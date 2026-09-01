"""
Tests for individual gates. Includes both probability-level checks and
phase-aware checks on the raw statevector amplitudes (probabilities
alone can't distinguish e.g. |1> from -|1> or i|1>, so several tests
here inspect `get_statevector().data` directly instead of just
`get_probabilities()`).
"""

import math

import pytest

from quantum_circuit_simulator import CircuitSimulator

SQRT2_INV = 1 / math.sqrt(2)


# ---------- X, Y, Z (probability-level) ----------

def test_x_gate_flips_zero_to_one():
    c = CircuitSimulator(1)
    c.x(0)
    probs = c.get_probabilities()
    assert probs == pytest.approx({"1": 1.0})


def test_x_gate_twice_returns_to_zero():
    c = CircuitSimulator(1)
    c.x(0).x(0)
    probs = c.get_probabilities()
    assert probs == pytest.approx({"0": 1.0})


def test_y_gate_flips_bit():
    c = CircuitSimulator(1)
    c.y(0)
    probs = c.get_probabilities()
    assert probs == pytest.approx({"1": 1.0})


def test_z_gate_leaves_zero_probability_unchanged():
    c = CircuitSimulator(1)
    c.z(0)
    probs = c.get_probabilities()
    assert probs == pytest.approx({"0": 1.0})


# ---------- phase-aware tests ----------

def test_z_gate_changes_relative_phase():
    """H|0> = (|0>+|1>)/sqrt2, then Z flips the sign of the |1> component:
    result should be (|0> - |1>)/sqrt2, not just 'still 50/50'."""
    c = CircuitSimulator(1)
    c.h(0).z(0)
    state = c.get_statevector().data
    expected = [SQRT2_INV, -SQRT2_INV]
    assert state == pytest.approx(expected)


def test_s_gate_applies_i_phase_to_one():
    """S|1> = i|1>: probabilities are unaffected (still 100% |1>), but
    the amplitude itself must pick up a factor of i."""
    c = CircuitSimulator(1)
    c.x(0).s(0)
    state = c.get_statevector().data
    expected = [0, 1j]
    assert state == pytest.approx(expected)


def test_t_gate_applies_eighth_turn_phase_to_one():
    """T|1> = e^(i*pi/4)|1>."""
    c = CircuitSimulator(1)
    c.x(0).t(0)
    state = c.get_statevector().data
    phase = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
    expected = [0, phase]
    assert state == pytest.approx(expected)


def test_rz_applies_expected_relative_phase():
    """Rz(theta)|1> = e^(i*theta/2)|1> (up to the qubit still being |1>
    with probability 1); verifies the amplitude's phase directly."""
    c = CircuitSimulator(1)
    theta = math.pi / 2
    c.x(0).rz(0, theta)
    state = c.get_statevector().data
    phase = complex(math.cos(theta / 2), math.sin(theta / 2))
    expected = [0, phase]
    assert state == pytest.approx(expected)


def test_two_s_gates_equal_one_z_gate():
    """S^2 = Z: applying S twice should match applying Z once, both in
    probability and in phase."""
    c_s = CircuitSimulator(1)
    c_s.h(0).s(0).s(0)

    c_z = CircuitSimulator(1)
    c_z.h(0).z(0)

    assert c_s.get_statevector().data == pytest.approx(c_z.get_statevector().data)


# ---------- H (superposition) ----------

def test_h_gate_creates_equal_superposition():
    c = CircuitSimulator(1)
    c.h(0)
    probs = c.get_probabilities()
    assert probs == pytest.approx({"0": 0.5, "1": 0.5})


def test_h_gate_twice_returns_to_zero():
    """H is self-inverse: H(H|0>) = |0>."""
    c = CircuitSimulator(1)
    c.h(0).h(0)
    state = c.get_statevector().data
    assert state == pytest.approx([1, 0])


# ---------- CNOT / entanglement ----------

def test_cx_without_superposition_stays_separable():
    """CNOT alone does not create entanglement: with the control in a
    definite state (not superposition), the result is a single, definite
    basis state, not a superposition."""
    c = CircuitSimulator(2)
    c.x(0).cx(0, 1)  # control=1 -> target flips: |10> -> |11>
    probs = c.get_probabilities()
    assert probs == pytest.approx({"11": 1.0})


def test_h_then_cx_creates_bell_state():
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    probs = c.get_probabilities()
    assert probs == pytest.approx({"00": 0.5, "11": 0.5})
    assert "01" not in probs
    assert "10" not in probs


# ---------- CZ ----------

def test_cz_flips_phase_only_when_both_are_one():
    c = CircuitSimulator(2)
    c.h(0).h(1).cz(0, 1)
    state = c.get_statevector().data
    # basis order (little-endian): 00, 01, 10, 11 -> only |11> picks up a minus sign
    expected = [0.5, 0.5, 0.5, -0.5]
    assert state == pytest.approx(expected)


# ---------- SWAP ----------

def test_swap_exchanges_qubit_states():
    c = CircuitSimulator(2)
    c.x(0).swap(0, 1)  # q0=1,q1=0 -> after swap q0=0,q1=1
    probs = c.get_probabilities()
    assert probs == pytest.approx({"10": 1.0})


# ---------- Toffoli ----------

def test_toffoli_flips_target_when_both_controls_are_one():
    c = CircuitSimulator(3)
    c.x(0).x(1).toffoli(0, 1, 2)
    probs = c.get_probabilities()
    assert probs == pytest.approx({"111": 1.0})


def test_toffoli_leaves_target_when_one_control_is_zero():
    c = CircuitSimulator(3)
    c.x(0).toffoli(0, 1, 2)  # control1=1, control2=0 -> target unaffected
    probs = c.get_probabilities()
    assert probs == pytest.approx({"001": 1.0})


# ---------- history / chaining ----------

def test_gate_methods_return_self_for_chaining():
    c = CircuitSimulator(2)
    result = c.h(0).cx(0, 1)
    assert result is c


def test_history_records_gates_in_order():
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    assert c.history == ["H(q0)", "CNOT(control=q0, target=q1)"]
