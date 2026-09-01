"""
Noise Models
=============
Configuration and construction of Qiskit Aer noise models, so
`CircuitSimulator` can simulate a circuit "as if" it were running on
imperfect, real hardware instead of an ideal (noiseless) simulator.

Four independent noise channels are supported, each with its own
probability (0.0 = off, 1.0 = maximum):

- bit_flip        : a single-qubit Pauli-X error applied to every gate
                     (models a qubit randomly flipping |0> <-> |1>)
- phase_flip       : a single-qubit Pauli-Z error applied to every gate
                     (models a qubit randomly losing phase coherence)
- depolarizing     : a single-qubit depolarizing channel applied to
                     every gate (models generic, non-specific noise
                     that scrambles the qubit's state)
- readout          : a symmetric readout (measurement) error — the
                     probability that a measured bit is reported
                     flipped from its true value

These are intentionally simplified, idealized noise channels (independent
per-qubit, identical across all qubits and gates). They are useful for
*demonstrating* how noise degrades results and by how much, not for
reproducing the specific noise characteristics of any particular real
quantum device.

Author: Yasmin Kasem
"""

import math
from dataclasses import dataclass
from numbers import Real

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error


def _validate_probability(name: str, value: Real) -> None:
    """
    Makes sure a noise probability is an actual real number between 0
    and 1 (inclusive). Rejects bool (since bool is technically an int
    subclass in Python, but True/False as a "probability" makes no
    sense here) and non-finite values like NaN or infinity.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


@dataclass(frozen=True)
class NoiseConfig:
    """
    Configuration for a noise model, as a set of independent
    probabilities. Defaults to 0.0 for every channel (i.e. no noise),
    so you only need to set the channels you actually want to enable.

    Example:
        NoiseConfig(bit_flip=0.01, readout=0.05)
    """

    bit_flip: float = 0.0
    phase_flip: float = 0.0
    depolarizing: float = 0.0
    readout: float = 0.0

    def __post_init__(self):
        _validate_probability("bit_flip", self.bit_flip)
        _validate_probability("phase_flip", self.phase_flip)
        _validate_probability("depolarizing", self.depolarizing)
        _validate_probability("readout", self.readout)

    def is_trivial(self) -> bool:
        """True if every channel is 0.0, i.e. this config represents no noise at all."""
        return (
            self.bit_flip == 0.0
            and self.phase_flip == 0.0
            and self.depolarizing == 0.0
            and self.readout == 0.0
        )


# Gates the per-gate (bit flip / phase flip / depolarizing) errors are
# attached to. Kept deliberately small and explicit rather than trying
# to catch "every possible gate name", since new gate methods added to
# CircuitSimulator should also update this list if they should be
# subject to noise.
_SINGLE_QUBIT_GATES = ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"]
_TWO_QUBIT_GATES = ["cx", "cz", "swap"]
_THREE_QUBIT_GATES = ["ccx"]


def build_noise_model(config: NoiseConfig) -> NoiseModel:
    """
    Builds a Qiskit Aer NoiseModel from a NoiseConfig.

    - bit_flip, phase_flip, and depolarizing are combined into a single
      per-qubit quantum error and attached to every single-qubit gate.
      For two- and three-qubit gates, that same per-qubit error model is
      expanded across each qubit involved in the operation using tensor
      products. This is a simplified, independent-error model and does
      not claim to represent the true error rate of real multi-qubit
      hardware gates.
    - readout is attached as a symmetric ReadoutError (same probability
      of a 0 being reported as 1 and a 1 being reported as 0) on every
      qubit.

    Passing a fully trivial NoiseConfig (all zeros) returns an empty
    NoiseModel, which is equivalent to running with no noise at all.
    """
    noise_model = NoiseModel()

    error_components = []
    if config.bit_flip > 0.0:
        error_components.append(
            pauli_error([("X", config.bit_flip), ("I", 1 - config.bit_flip)])
        )
    if config.phase_flip > 0.0:
        error_components.append(
            pauli_error([("Z", config.phase_flip), ("I", 1 - config.phase_flip)])
        )
    if config.depolarizing > 0.0:
        error_components.append(depolarizing_error(config.depolarizing, 1))

    if error_components:
        combined_error = error_components[0]
        for component in error_components[1:]:
            combined_error = combined_error.compose(component)

        noise_model.add_all_qubit_quantum_error(combined_error, _SINGLE_QUBIT_GATES)

        # Two- and three-qubit gates touch more qubits, so we attach an
        # expanded (tensor) version of the same single-qubit error to
        # each qubit the gate acts on. This does not increase the
        # per-qubit error probability itself - it simply applies the
        # same independent per-qubit error model to every qubit involved
        # in the multi-qubit operation. It is a simplification and is
        # not intended to reproduce the true error rate of real
        # multi-qubit hardware gates (which are typically noisier than
        # this independent-error model implies).
        two_qubit_error = combined_error.tensor(combined_error)
        noise_model.add_all_qubit_quantum_error(two_qubit_error, _TWO_QUBIT_GATES)

        three_qubit_error = two_qubit_error.tensor(combined_error)
        noise_model.add_all_qubit_quantum_error(three_qubit_error, _THREE_QUBIT_GATES)

    if config.readout > 0.0:
        p = config.readout
        readout_error = ReadoutError([[1 - p, p], [p, 1 - p]])
        noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model
