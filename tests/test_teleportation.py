"""
Tests for the quantum teleportation example. Covers both amplitude
(theta) and relative phase (phi), since a state with the wrong phase can
still produce the right computational-basis *probabilities* - so
teleportation needs to be verified with a non-trivial phi too, not just
a range of theta values, to prove it isn't accidentally only correct
for real-amplitude states.
"""

import math

import pytest

from examples.teleportation import (
    quantum_teleportation_demo,
    teleportation_multi_state_test,
)


@pytest.mark.parametrize(
    "theta,phi",
    [
        (0.0, 0.0),
        (math.pi, 0.0),
        (math.pi / 4, 0.0),
        (math.pi / 2, 0.0),
        (math.pi / 2, math.pi / 2),
        (math.pi / 3, math.pi),
        (2.1, 4.2),
    ],
)
def test_teleportation_succeeds_for_state(theta, phi):
    result = quantum_teleportation_demo(theta=theta, phi=phi, verbose=False)
    assert result["success_rate"] > 99.0


def test_teleportation_multi_state_all_pass():
    results = teleportation_multi_state_test(
        states=[
            (0.0, 0.0),
            (math.pi / 4, 0.0),
            (math.pi / 2, math.pi / 2),
            (math.pi / 3, math.pi),
        ]
    )
    assert all(r["success_rate"] > 99.0 for r in results)


def test_teleportation_result_contains_theta_and_phi():
    result = quantum_teleportation_demo(theta=1.0, phi=0.5, verbose=False)
    assert result["theta"] == 1.0
    assert result["phi"] == 0.5
