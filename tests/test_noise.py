"""
Tests for the noise module (NoiseConfig validation, build_noise_model)
and CircuitSimulator's noise-aware methods (run_measurement with a
noise_model, compare_noise_impact).

Statistical assertions use a fixed `seed_simulator` and a deliberately
strong noise probability, so the tests are reproducible and the effect
of noise is large enough to be unambiguous rather than relying on
`noise_contribution > 0`, which could occasionally be flaky at small,
realistic noise levels.
"""

import pytest

from quantum_circuit_simulator import CircuitSimulator, NoiseConfig, build_noise_model

# ---------- NoiseConfig validation ----------

def test_noise_config_defaults_to_no_noise():
    config = NoiseConfig()
    assert config.is_trivial()


def test_noise_config_probability_out_of_range_raises():
    with pytest.raises(ValueError):
        NoiseConfig(bit_flip=1.5)


def test_noise_config_negative_probability_raises():
    with pytest.raises(ValueError):
        NoiseConfig(readout=-0.1)


def test_noise_config_bool_raises():
    with pytest.raises(TypeError):
        NoiseConfig(depolarizing=True)


def test_noise_config_non_numeric_raises():
    with pytest.raises(TypeError):
        NoiseConfig(phase_flip="high")


def test_noise_config_nan_raises():
    with pytest.raises(ValueError):
        NoiseConfig(bit_flip=float("nan"))


def test_noise_config_boundary_values_ok():
    config = NoiseConfig(bit_flip=0.0, phase_flip=1.0, depolarizing=0.5, readout=1.0)
    assert config.readout == 1.0


# ---------- build_noise_model ----------

def test_build_noise_model_trivial_config_has_no_errors():
    noise_model = build_noise_model(NoiseConfig())
    assert noise_model.to_dict()["errors"] == []


def test_build_noise_model_readout_only():
    noise_model = build_noise_model(NoiseConfig(readout=0.1))
    errors = noise_model.to_dict()["errors"]
    assert any(e["type"] == "roerror" for e in errors)


def test_build_noise_model_gate_errors_present():
    noise_model = build_noise_model(NoiseConfig(bit_flip=0.05))
    errors = noise_model.to_dict()["errors"]
    assert any(e["type"] == "qerror" for e in errors)


# ---------- noisy measurement (statistical, seeded) ----------

def test_readout_noise_flips_a_deterministic_outcome():
    """With a very strong readout error (p=1.0, i.e. always flip the
    reported bit) on a deterministic |1> state, every measurement
    should be reported as 0 instead of 1."""
    c = CircuitSimulator(1)
    c.x(0)
    noise_model = build_noise_model(NoiseConfig(readout=1.0))
    counts = c.run_measurement(shots=500, noise_model=noise_model, seed_simulator=1)
    assert counts == {"0": 500}


def test_noise_model_none_means_no_noise():
    """Passing noise_model=None (the default) should give identical
    results to not passing it at all, for the same seed."""
    c1 = CircuitSimulator(2)
    c1.h(0).cx(0, 1)
    counts1 = c1.run_measurement(shots=1000, seed_simulator=3)

    c2 = CircuitSimulator(2)
    c2.h(0).cx(0, 1)
    counts2 = c2.run_measurement(shots=1000, noise_model=None, seed_simulator=3)

    assert counts1 == counts2


def test_compare_noise_impact_returns_expected_keys():
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    noise_model = build_noise_model(NoiseConfig(readout=0.2))
    result = c.compare_noise_impact(noise_model, shots=4096, seed_simulator=42)
    for key in (
        "theoretical",
        "ideal_measured",
        "noisy_measured",
        "tvd_sampling_only",
        "tvd_sampling_and_noise",
        "observed_noise_delta",
    ):
        assert key in result


def test_compare_noise_impact_shows_noise_increases_distance():
    """With a deliberately strong readout error (p=0.3, well above
    ordinary sampling noise at 4096 shots), the noisy TVD should be
    clearly larger than the noiseless TVD."""
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    noise_model = build_noise_model(NoiseConfig(readout=0.3))
    result = c.compare_noise_impact(noise_model, shots=4096, seed_simulator=42)
    assert result["tvd_sampling_and_noise"] > result["tvd_sampling_only"]
    assert result["observed_noise_delta"] > 0.1


def test_compare_noise_impact_zero_noise_config_close_to_ideal():
    """A trivial (all-zero) NoiseConfig should behave like no noise at
    all - the observed delta should be small."""
    c = CircuitSimulator(2)
    c.h(0).cx(0, 1)
    noise_model = build_noise_model(NoiseConfig())
    result = c.compare_noise_impact(noise_model, shots=4096, seed_simulator=42)
    assert abs(result["observed_noise_delta"]) < 0.05
