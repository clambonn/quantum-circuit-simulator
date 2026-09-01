# Quantum Circuit Simulator

[![Tests](https://github.com/clambonn/quantum-circuit-simulator/actions/workflows/tests.yml/badge.svg)](https://github.com/clambonn/quantum-circuit-simulator/actions/workflows/tests.yml)

A small Python package for building, simulating, and analyzing quantum
circuits, built on top of **Qiskit** and **Qiskit Aer**.

> **Scope note:** This project does *not* implement a quantum
> mechanics simulator from scratch. Qiskit and Qiskit Aer perform the
> actual statevector calculations and sampling. What this project
> provides is a validated, chainable interface on top of them — plus
> additional analysis (theoretical vs. measured probability
> comparison, noise modeling, statistical distance metrics,
> benchmarking) and worked examples (Bell state, GHZ state, quantum
> teleportation) that demonstrate core quantum computing concepts end
> to end.

## Why this project

Most "Qiskit wrapper" projects stop at building a circuit and printing
a histogram. This one goes further:

- Every gate call is **validated** (qubit range, distinct
  control/target qubits, correct types, finite rotation angles) with
  clear error messages.
- Circuit output includes both the **theoretical statevector**
  (computed analytically, no sampling) and the **measured
  distribution** (sampled via `shots` on `AerSimulator`), with a
  built-in comparison using Mean Absolute Error, Max Absolute Error,
  and Total Variation Distance.
- **Noise simulation**: bit flip, phase flip, depolarizing, and
  readout error channels, so you can compare an ideal circuit against
  a simulated imperfect one and see exactly how much the results
  degrade.
- **Benchmarking**: measures how statevector simulation time grows
  with the number of qubits, and plots it against the 2ⁿ amplitude
  count to make the exponential cost visible.
- A full **quantum teleportation** circuit is implemented with
  mid-circuit measurement and classical control (`if_test`), verified
  across multiple input states — including relative phase, not just
  probability — not just one.
- A `pytest` test suite (100+ tests) covers validation, individual
  gates (including phase, not just probabilities), known entangled
  states, measurement, noise, quantum teleportation, and the
  distribution-comparison statistics themselves (`analysis.py`), and
  runs automatically on every push via GitHub Actions.

## Project structure

```
quantum-circuit-simulator/
│
├── src/
│   └── quantum_circuit_simulator/
│       ├── __init__.py        # public API
│       ├── simulator.py       # CircuitSimulator: gates, validation, measurement
│       ├── noise.py           # NoiseConfig + build_noise_model
│       ├── analysis.py        # TVD / MAE / probability comparison (pure functions)
│       └── visualization.py   # histogram + benchmark + noise-sweep plotting
│
├── examples/
│   ├── bell_state.py
│   ├── ghz_state.py
│   ├── teleportation.py       # quantum teleportation demonstration (self-contained)
│   ├── noise_simulation.py    # ideal vs. noisy vs. measured + readout error sweep
│   └── benchmark.py           # simulation time vs. number of qubits
│
├── tests/
│   ├── test_validation.py
│   ├── test_gates.py
│   ├── test_measurement.py
│   ├── test_noise.py
│   ├── test_teleportation.py
│   └── test_analysis.py       # TVD / MAE / probability comparison, tested standalone
│
├── .github/workflows/tests.yml  # CI: runs pytest + ruff on Python 3.10-3.12
├── pyproject.toml
├── LICENSE
├── .gitignore
└── README.md
```

## Installation

```bash
git clone <https://github.com/clambonn/quantum-circuit-simulator>
cd quantum-circuit-simulator
pip install -e ".[dev]"
```

## Usage

```python
from quantum_circuit_simulator import CircuitSimulator, NoiseConfig, build_noise_model

# Build a Bell state (an entangled pair of qubits)
bell = CircuitSimulator(num_qubits=2, name="bell_state")
bell.h(0).cx(0, 1)                       # chainable gate calls

bell.show_circuit()                      # ASCII circuit diagram
bell.show_statevector()                  # exact theoretical probabilities
bell.show_comparison(shots=4096)         # theory vs. sampled results

# Simulate noisy hardware
noise_model = build_noise_model(NoiseConfig(bit_flip=0.01, readout=0.05))
noisy_counts = bell.run_measurement(shots=4096, noise_model=noise_model)
```

Run the worked examples directly:

```bash
python examples/bell_state.py
python examples/ghz_state.py
python examples/teleportation.py
python examples/noise_simulation.py
python examples/benchmark.py
```

## Supported gates

| Category      | Gates                                  |
|----------------|-----------------------------------------|
| Single-qubit   | `X`, `Y`, `Z`, `H`, `S`, `T`, `Rx`, `Ry`, `Rz` |
| Two-qubit      | `CNOT` (`cx`), `CZ` (`cz`), `SWAP`      |
| Three-qubit    | `Toffoli` (`ccx`)                       |

## Featured concepts

### Bell state — entanglement in its simplest form
`H(q0)` puts qubit 0 in superposition, then `CNOT(q0, q1)` entangles
it with qubit 1. The result: 50% `|00⟩`, 50% `|11⟩`, and *never*
`|01⟩` or `|10⟩`. Measuring one qubit instantly tells you the other.

Important nuance verified in this project: **CNOT alone does not
create entanglement.** If the control qubit is in a definite state
(`|0⟩` or `|1⟩`, not a superposition), CNOT leaves the two qubits
separable. Entanglement requires the control to be in superposition
first (see `test_cx_without_superposition_stays_separable` in
`tests/test_gates.py`).

### GHZ state
The same idea generalized to three qubits: `H(q0)` + `CNOT(0,1)` +
`CNOT(1,2)` produces 50% `|000⟩`, 50% `|111⟩`.

### Quantum teleportation
Transfers an arbitrary quantum state from one qubit to another using
a shared entangled pair and two classical bits — without moving the
physical qubit itself. Implemented with genuine mid-circuit
measurement and classically-controlled gates (`qc.if_test(...)`).

The state to be teleported is prepared with `Ry(theta)` **and**
`Rz(phi)`, and verified with both parameters across multiple pairs —
not just amplitude (`theta`), but relative **phase** (`phi`) too,
since a state with the wrong phase can still measure the right
computational-basis probabilities. Probability-only verification
would not have been enough to prove teleportation is fully correct.

> Note: the original **quantum state** is destroyed by measurement
> during teleportation — this is required by the no-cloning theorem.
> The **physical qubit** itself is not destroyed; it simply no longer
> holds the teleported state.

### Noise simulation
`examples/noise_simulation.py` models four independent noise channels
via `NoiseConfig`:

| Channel        | Models                                             |
|----------------|-----------------------------------------------------|
| `bit_flip`     | a qubit randomly flipping `|0⟩ ↔ |1⟩`               |
| `phase_flip`   | a qubit randomly losing phase coherence             |
| `depolarizing` | generic, non-specific noise scrambling the state    |
| `readout`      | the reported measurement bit being flipped          |

It compares three distributions side by side — theoretical (exact),
noiseless-measured (sampling noise only), and noisy-measured (sampling
noise + simulated hardware noise) — and sweeps the readout error
probability to show Total Variation Distance increasing as noise
increases.

These are simplified, idealized channels (independent per qubit,
identical across all qubits/gates) meant to *demonstrate* how noise
degrades results, not to reproduce the specific noise profile of any
particular real quantum device.

### Benchmark: simulation cost vs. number of qubits
`examples/benchmark.py` times exact statevector computation across an
increasing number of qubits and plots it (log scale) against qubit
count. A statevector stores one complex amplitude per basis state, and
an *n*-qubit system has 2ⁿ basis states:

```
10 qubits →         1,024 amplitudes
20 qubits →     1,048,576 amplitudes
30 qubits → 1,073,741,824 amplitudes
```

The statevector representation requires 2ⁿ complex amplitudes, so
**memory requirements grow exponentially** with qubit count. Runtime
generally becomes increasingly expensive for the same reason, though
measured timing also depends on the simulator's implementation and the
hardware it runs on — this is the same underlying reason full classical
simulation of quantum computers doesn't scale indefinitely, and part of
why quantum computers are interesting in the first place. Circuit
construction is excluded from the timed portion, and each width is
timed 5 times with the median taken, to reduce noise in the measurement
itself.

## Testing

```bash
pytest -v
```

or, for a coverage report:

```bash
pytest --cov=quantum_circuit_simulator --cov-report=term-missing
```

The test suite covers:
- Input validation (`num_qubits`, qubit indices, distinct
  control/target, rotation angles, `shots`, `qubits` parameter types)
- Individual gate correctness, including **phase-aware** checks on raw
  statevector amplitudes (`Z`, `S`, `T`, `Rz`), not just probabilities
- Bell state and GHZ state probabilities
- Full and partial measurement, with fixed `seed_simulator` values for
  reproducibility
- Noise model construction/validation and its statistical effect on
  measured distributions
- Theoretical vs. measured statistical comparison
- Quantum teleportation across multiple `(theta, phi)` pairs
- The pure statistics functions in `analysis.py` (`counts_to_probabilities`,
  `total_variation_distance`, `mean_absolute_error`, `max_absolute_error`,
  `compare_distributions`) tested independently against hand-computed
  distributions (identical, disjoint, partially overlapping), including
  input validation for empty/negative/non-integer counts

CI runs the full suite (plus `ruff` linting) on Python 3.10, 3.11, and
3.12 via GitHub Actions on every push and pull request
(`.github/workflows/tests.yml`).

## Statistical comparison methodology

`compare_theoretical_vs_measured()` computes:

- **Total Absolute Error** — sum of `|theoretical − measured|` across
  all outcomes (not normalized by outcome count).
- **Mean Absolute Error (MAE)** — average per-outcome difference.
- **Max Absolute Error** — largest single-outcome discrepancy.
- **Total Variation Distance (TVD)** — `0.5 × Σ|p − q|`, the standard
  metric for comparing two probability distributions. Ranges from 0
  (identical) to 1 (completely different) and is independent of the
  number of outcomes, making it more meaningful than the raw total
  error for cross-circuit comparisons.

**On interpretation:** this quantifies the statistical difference
between the exact probabilities (from the statevector) and the
sampled measurement results (from `shots` runs on the simulator) — it
is not a measurement of simulator *accuracy*. Small, non-zero
differences here are expected and shrink as `shots` grows; they
reflect ordinary statistical sampling noise inherent to measurement,
not an error in Qiskit's simulation.

`compare_noise_impact()` extends this to compare a noiseless run
against a noisy one, reporting an `observed_noise_delta` — the
*empirical* change in TVD between the two experiments for that
specific circuit and noise model. This is not an exact decomposition
of "how much error belongs to noise," since sampling noise and
hardware noise aren't independent, additive quantities — it's simply
the difference actually observed.

## Author

Yasmin Kasem
