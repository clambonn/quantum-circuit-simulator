"""
quantum_circuit_simulator
===========================
A small educational toolkit for building, simulating, and analyzing
quantum circuits with Qiskit and Qiskit Aer.

Public API:
    CircuitSimulator      - build/run/measure circuits
    NoiseConfig           - configure a noise model (bit flip, phase
                             flip, depolarizing, readout error)
    build_noise_model     - turn a NoiseConfig into a qiskit_aer NoiseModel
    analysis              - distribution-comparison functions (TVD, MAE, ...)
    visualization          - plotting/saving helpers
"""

from . import analysis, visualization
from .noise import NoiseConfig, build_noise_model
from .simulator import CircuitSimulator

__all__ = [
    "CircuitSimulator",
    "NoiseConfig",
    "analysis",
    "build_noise_model",
    "visualization",
]

__version__ = "0.1.0"
