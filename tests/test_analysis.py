"""
Tests for the pure, side-effect-free statistics helpers in analysis.py
(counts_to_probabilities, total_variation_distance, mean_absolute_error,
max_absolute_error, compare_distributions). These are tested completely
independently of CircuitSimulator/Qiskit, using known distributions with
hand-computed expected results.
"""

import pytest

from quantum_circuit_simulator.analysis import (
    compare_distributions,
    counts_to_probabilities,
    max_absolute_error,
    mean_absolute_error,
    total_variation_distance,
)

# ---------- counts_to_probabilities ----------


def test_counts_to_probabilities_basic():
    counts = {"00": 512, "11": 488}
    probs = counts_to_probabilities(counts)
    assert probs["00"] == pytest.approx(512 / 1000)
    assert probs["11"] == pytest.approx(488 / 1000)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_counts_to_probabilities_single_outcome():
    probs = counts_to_probabilities({"0": 100})
    assert probs == {"0": 1.0}


def test_counts_to_probabilities_empty_dict_raises():
    with pytest.raises(ValueError):
        counts_to_probabilities({})


def test_counts_to_probabilities_zero_total_raises():
    with pytest.raises(ValueError):
        counts_to_probabilities({"0": 0, "1": 0})


def test_counts_to_probabilities_negative_count_raises():
    with pytest.raises(ValueError):
        counts_to_probabilities({"00": -10, "11": 20})


def test_counts_to_probabilities_bool_count_raises():
    with pytest.raises(ValueError):
        counts_to_probabilities({"0": True, "1": 5})


def test_counts_to_probabilities_non_int_count_raises():
    with pytest.raises(ValueError):
        counts_to_probabilities({"0": 1.5, "1": 2.5})


# ---------- total_variation_distance ----------


def test_tvd_identical_distributions_is_zero():
    a = {"0": 0.5, "1": 0.5}
    b = {"0": 0.5, "1": 0.5}
    assert total_variation_distance(a, b) == pytest.approx(0.0)


def test_tvd_disjoint_distributions_is_one():
    a = {"0": 1.0}
    b = {"1": 1.0}
    assert total_variation_distance(a, b) == pytest.approx(1.0)


def test_tvd_partial_overlap():
    # Half the probability mass disagrees -> TVD = 0.5 * (0.5 + 0.5) = 0.5
    a = {"0": 1.0, "1": 0.0}
    b = {"0": 0.5, "1": 0.5}
    assert total_variation_distance(a, b) == pytest.approx(0.5)


def test_tvd_is_symmetric():
    a = {"0": 0.7, "1": 0.3}
    b = {"0": 0.4, "1": 0.6}
    assert total_variation_distance(a, b) == pytest.approx(total_variation_distance(b, a))


def test_tvd_missing_outcomes_treated_as_zero():
    a = {"00": 1.0}
    b = {"00": 0.5, "11": 0.5}
    # |1.0-0.5| + |0.0-0.5| = 1.0, TVD = 0.5
    assert total_variation_distance(a, b) == pytest.approx(0.5)


# ---------- mean_absolute_error / max_absolute_error ----------


def test_mae_identical_distributions_is_zero():
    a = {"0": 0.5, "1": 0.5}
    assert mean_absolute_error(a, a) == pytest.approx(0.0)


def test_mae_basic():
    a = {"0": 1.0, "1": 0.0}
    b = {"0": 0.5, "1": 0.5}
    # errors: |1-0.5|=0.5, |0-0.5|=0.5 -> mean = 0.5
    assert mean_absolute_error(a, b) == pytest.approx(0.5)


def test_max_absolute_error_basic():
    a = {"0": 1.0, "1": 0.0}
    b = {"0": 0.6, "1": 0.4}
    # errors: |1-0.6|=0.4, |0-0.4|=0.4 -> max = 0.4
    assert max_absolute_error(a, b) == pytest.approx(0.4)


def test_max_absolute_error_picks_largest_single_outcome():
    a = {"00": 0.9, "01": 0.1, "10": 0.0}
    b = {"00": 0.5, "01": 0.1, "10": 0.4}
    # errors: 0.4, 0.0, 0.4 -> max = 0.4
    assert max_absolute_error(a, b) == pytest.approx(0.4)


def test_mae_and_max_empty_distributions_are_zero():
    assert mean_absolute_error({}, {}) == 0.0
    assert max_absolute_error({}, {}) == 0.0


# ---------- compare_distributions ----------


def test_compare_distributions_identical():
    a = {"0": 0.5, "1": 0.5}
    result = compare_distributions(a, a)
    assert result["total_absolute_error"] == pytest.approx(0.0)
    assert result["mean_absolute_error"] == pytest.approx(0.0)
    assert result["max_absolute_error"] == pytest.approx(0.0)
    assert result["total_variation_distance"] == pytest.approx(0.0)
    assert result["distribution_a"] == a
    assert result["distribution_b"] == a


def test_compare_distributions_disjoint():
    a = {"0": 1.0}
    b = {"1": 1.0}
    result = compare_distributions(a, b)
    assert result["total_variation_distance"] == pytest.approx(1.0)
    assert result["total_absolute_error"] == pytest.approx(2.0)
    assert result["max_absolute_error"] == pytest.approx(1.0)


def test_compare_distributions_keys_present():
    result = compare_distributions({"0": 1.0}, {"0": 0.9, "1": 0.1})
    expected_keys = {
        "distribution_a",
        "distribution_b",
        "total_absolute_error",
        "mean_absolute_error",
        "max_absolute_error",
        "total_variation_distance",
    }
    assert expected_keys.issubset(result.keys())
