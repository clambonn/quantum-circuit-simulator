"""
Quantum Circuit Simulator
===========================
A simplified interface for building and simulating quantum circuits using Qiskit and Aer.
(All the actual simulation and computation is done by Qiskit — this class is
just a thinner layer on top of it that makes it easy to add gates, with
validation, optional noise, and comparison between distributions).

Lets you build a circuit by adding gates (X, H, CNOT, Rx, ...) across a
number of qubits, then run the circuit and get back: the statevector,
the probabilities, and a histogram of the measurement results.

This class only ever returns data (dicts, Statevector objects, counts).
Printing lives in the thin `show_*` helpers; plotting lives in
`visualization.py`; distribution comparison math lives in `analysis.py`.

Author: Yasmin Kasem
"""

import math
from collections.abc import Sequence
from numbers import Real

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .analysis import compare_distributions, counts_to_probabilities
from .visualization import save_histogram


class CircuitSimulator:
    """
    A class that wraps Qiskit's QuantumCircuit and adds a simpler interface
    for adding gates, with validation, running the circuit, and displaying results.
    """

    def __init__(self, num_qubits: int, name: str = "circuit"):
        # Validation: num_qubits must be a positive integer, otherwise there's
        # no physical meaning to creating a circuit with it (0 qubits or -2 qubits, for example)
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
            raise TypeError(f"num_qubits must be an int, got {type(num_qubits).__name__}")
        if num_qubits <= 0:
            raise ValueError(f"num_qubits must be greater than 0, got {num_qubits}")

        self.num_qubits = num_qubits
        self.qc = QuantumCircuit(num_qubits, name=name)
        self.history = []  # keeps track of every gate added, in order

    # =====================================================
    # Validation helpers
    # =====================================================
    def _check_qubit(self, qubit: int):
        """
        Makes sure the qubit index actually exists in the circuit.
        For example, if you have 3 qubits (numbered 0,1,2) and try to use
        qubit number 5, the code stops and gives you a clear message instead
        of letting an obscure error happen deep inside Qiskit.
        """
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise TypeError(f"Qubit index must be an int, got {type(qubit).__name__}")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(
                f"Qubit {qubit} out of range. "
                f"This circuit has {self.num_qubits} qubits (valid range: 0-{self.num_qubits - 1})"
            )

    def _check_distinct(self, q1: int, q2: int, label1="control", label2="target"):
        """
        Makes sure two qubits (like control and target in CNOT) aren't the
        same qubit. Physically, something like CNOT(0, 0) doesn't make sense.
        """
        if q1 == q2:
            raise ValueError(
                f"{label1} and {label2} must be different qubits, "
                f"but both were q{q1}"
            )

    def _check_angle(self, theta: Real):
        """
        Makes sure a rotation angle (for Rx/Ry/Rz) is an actual finite
        real number. Rejects bool, strings, complex numbers, NaN, and
        +-infinity, all of which would otherwise fail deep inside
        Qiskit with a much less clear error message.
        """
        if isinstance(theta, bool) or not isinstance(theta, Real):
            raise TypeError(f"theta must be a real number, got {type(theta).__name__}")
        if not math.isfinite(theta):
            raise ValueError(f"theta must be finite, got {theta}")

    def _validate_qubits(self, qubits: Sequence[int]):
        """
        Validates a list of qubits passed for partial measurement:
        every index must be in range, and none may repeat (measuring
        the same qubit twice into two classical bits doesn't make sense).
        """
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"qubits must not contain duplicates, got {qubits}")
        for q in qubits:
            self._check_qubit(q)

    # =====================================================
    # Single-qubit gates
    # =====================================================
    def x(self, qubit: int):
        """NOT gate: flips |0> to |1> and vice versa"""
        self._check_qubit(qubit)
        self.qc.x(qubit)
        self.history.append(f"X(q{qubit})")
        return self

    def y(self, qubit: int):
        """Y gate: matrix [[0,-i],[i,0]]. Flips the bit (like X) while also
        adding an imaginary phase.
        Mathematically: Y = iXZ (i.e. applying Y is equivalent to applying
        Z then X, multiplied by i as a global phase). That's why we say
        Y = i·X·Z rather than a general "combination" — the relationship
        is exact."""
        self._check_qubit(qubit)
        self.qc.y(qubit)
        self.history.append(f"Y(q{qubit})")
        return self

    def z(self, qubit: int):
        """Z gate: flips the phase of |1> (from + to -) and leaves |0> unchanged"""
        self._check_qubit(qubit)
        self.qc.z(qubit)
        self.history.append(f"Z(q{qubit})")
        return self

    def h(self, qubit: int):
        """Hadamard gate: puts the qubit into superposition
        (50% chance of |0> and 50% chance of |1>)"""
        self._check_qubit(qubit)
        self.qc.h(qubit)
        self.history.append(f"H(q{qubit})")
        return self

    def s(self, qubit: int):
        """S gate (Phase gate): rotates the phase by 90 degrees (pi/2).
        Like Z but half the angle. Important for building other, more complex gates."""
        self._check_qubit(qubit)
        self.qc.s(qubit)
        self.history.append(f"S(q{qubit})")
        return self

    def t(self, qubit: int):
        """T gate: rotates the phase by 45 degrees (pi/4).
        Very important in practice because it's one of the gates that's
        easy to implement physically on real hardware, and it's used in
        fault-tolerant quantum computing."""
        self._check_qubit(qubit)
        self.qc.t(qubit)
        self.history.append(f"T(q{qubit})")
        return self

    def rx(self, qubit: int, theta: float):
        """Rotation around the X axis by angle theta (in radians).
        This is a general (parametrized) gate — the X gate is a special
        case of it when theta = pi"""
        self._check_qubit(qubit)
        self._check_angle(theta)
        self.qc.rx(theta, qubit)
        self.history.append(f"Rx(q{qubit}, theta={theta:.4f})")
        return self

    def ry(self, qubit: int, theta: float):
        """Rotation around the Y axis by angle theta (in radians)"""
        self._check_qubit(qubit)
        self._check_angle(theta)
        self.qc.ry(theta, qubit)
        self.history.append(f"Ry(q{qubit}, theta={theta:.4f})")
        return self

    def rz(self, qubit: int, theta: float):
        """Rotation around the Z axis by angle theta (in radians)"""
        self._check_qubit(qubit)
        self._check_angle(theta)
        self.qc.rz(theta, qubit)
        self.history.append(f"Rz(q{qubit}, theta={theta:.4f})")
        return self

    # =====================================================
    # Two-qubit gates
    # =====================================================
    def cx(self, control: int, target: int):
        """CNOT gate: if the control qubit = |1>, applies X to the target.

        Important note: CNOT by itself doesn't automatically create
        entanglement. If the control qubit is in a "definite" state like
        |0> or |1> (not superposition), the result stays separable (e.g.
        |00> -> |00>, with no entanglement at all). Entanglement only
        happens when the control is first put into superposition (usually
        via H), and only then does CNOT link the qubits together. Example:
        H(q0) followed by CNOT(q0,q1) produces an actually entangled Bell
        state."""
        self._check_qubit(control)
        self._check_qubit(target)
        self._check_distinct(control, target)
        self.qc.cx(control, target)
        self.history.append(f"CNOT(control=q{control}, target=q{target})")
        return self

    def cz(self, control: int, target: int):
        """Controlled-Z: flips the phase if both qubits = |1>"""
        self._check_qubit(control)
        self._check_qubit(target)
        self._check_distinct(control, target)
        self.qc.cz(control, target)
        self.history.append(f"CZ(q{control}, q{target})")
        return self

    def swap(self, q1: int, q2: int):
        """SWAP gate: fully exchanges the state between one qubit and the other"""
        self._check_qubit(q1)
        self._check_qubit(q2)
        self._check_distinct(q1, q2, "q1", "q2")
        self.qc.swap(q1, q2)
        self.history.append(f"SWAP(q{q1}, q{q2})")
        return self

    # =====================================================
    # Three-qubit gates
    # =====================================================
    def toffoli(self, control1: int, control2: int, target: int):
        """Toffoli gate (CCX): like CNOT but with two control qubits.
        Applies X to the target only if both controls = |1>.
        Important because it's universal for reversible classical
        computation (meaning you can use it to build any reversible
        classical circuit, like AND/OR/NOT, but you need enough ancilla
        qubits set up appropriately — it doesn't cover every classical
        computation directly on its own)"""
        self._check_qubit(control1)
        self._check_qubit(control2)
        self._check_qubit(target)
        if len({control1, control2, target}) < 3:
            raise ValueError(
                f"control1 (q{control1}), control2 (q{control2}), and target (q{target}) "
                "must all be different qubits"
            )
        self.qc.ccx(control1, control2, target)
        self.history.append(f"Toffoli(c1=q{control1}, c2=q{control2}, target=q{target})")
        return self

    # =====================================================
    # Data (no printing, no plotting)
    # =====================================================
    def get_statevector(self) -> Statevector:
        """
        Computes the final statevector (the full mathematical description
        of the quantum state) without performing a measurement, so we can
        inspect the amplitudes and phases
        """
        return Statevector.from_instruction(self.qc)

    def get_probabilities(self) -> dict[str, float]:
        """The exact, analytically-computed probability of each outcome (no sampling)."""
        return self.get_statevector().probabilities_dict()

    def _build_measured_circuit(self, qubits: list[int] | None) -> QuantumCircuit:
        measured_qc = self.qc.copy()
        if qubits is None:
            measured_qc.measure_all()
        else:
            self._validate_qubits(qubits)
            measured_qc.add_register(ClassicalRegister(len(qubits)))
            for i, q in enumerate(qubits):
                measured_qc.measure(q, i)
        return measured_qc

    def run_measurement(
        self,
        shots: int = 1024,
        qubits: list[int] | None = None,
        noise_model: NoiseModel | None = None,
        seed_simulator: int | None = None,
    ) -> dict[str, int]:
        """
        Runs the circuit `shots` times on the simulator (similar to how it
        would run on real hardware), and returns the distribution of
        results (counts). Pure data - no printing, no plotting (use
        `show_measurement` or `visualization.save_histogram` for that).

        You can specify either:
          - qubits=None  -> measure_all() (measures every qubit)
          - qubits=[0,2] -> measures only these qubits (partial measurement)

        `noise_model`: an optional qiskit_aer NoiseModel (e.g. built with
        `noise.build_noise_model`) to simulate imperfect hardware. Left
        as None (the default), the simulation is ideal/noiseless -
        the only source of variation is ordinary statistical sampling.

        `seed_simulator`: an optional integer seed. Passing the same
        seed with the same circuit/shots/noise_model reproduces the
        exact same counts, which is useful for deterministic tests.
        """
        if not isinstance(shots, int):
            raise TypeError(f"shots must be an int, got {type(shots).__name__}")
        if shots <= 0:
            raise ValueError(f"shots must be greater than 0, got {shots}")

        if qubits is not None:
            if not isinstance(qubits, (list, tuple)):
                raise TypeError(
                    f"qubits must be a list or tuple, got {type(qubits).__name__}"
                )
            if len(qubits) == 0:
                raise ValueError(
                    "qubits list cannot be empty (use qubits=None to measure all)"
                )

        if noise_model is not None and not isinstance(noise_model, NoiseModel):
            raise TypeError(
                f"noise_model must be a NoiseModel or None, got {type(noise_model).__name__}"
            )

        measured_qc = self._build_measured_circuit(qubits)

        simulator = AerSimulator(noise_model=noise_model)
        job = simulator.run(measured_qc, shots=shots, seed_simulator=seed_simulator)
        counts = job.result().get_counts()
        return counts

    # =====================================================
    # Comparison (delegates the math to analysis.py)
    # =====================================================
    def compare_theoretical_vs_measured(
        self,
        shots: int = 4096,
        seed_simulator: int | None = None,
    ) -> dict[str, object]:
        """
        Compares the theoretical probabilities (from the statevector, no
        sampling) with the measured probabilities (from `shots` samples
        on an ideal/noiseless AerSimulator), and quantifies the
        statistical difference between them using several distance
        metrics (see `analysis.compare_distributions`).

        Note on interpretation: this quantifies the statistical
        difference between exact probabilities and sampled measurement
        results (ordinary sampling noise) - it is not a measurement of
        "simulator accuracy". Small, non-zero differences here are
        expected and should shrink as `shots` grows; they do not
        indicate that Qiskit's simulation itself is wrong.
        """
        theoretical = self.get_probabilities()
        measured_counts = self.run_measurement(shots=shots, seed_simulator=seed_simulator)
        measured = counts_to_probabilities(measured_counts)

        result = compare_distributions(theoretical, measured)
        # Keep the historical key names ("theoretical"/"measured") for
        # backwards compatibility with existing callers/tests.
        result["theoretical"] = result.pop("distribution_a")
        result["measured"] = result.pop("distribution_b")
        return result

    def compare_noise_impact(
        self,
        noise_model: NoiseModel,
        shots: int = 4096,
        seed_simulator: int | None = None,
    ) -> dict[str, object]:
        """
        Runs the same circuit twice with the same number of shots - once
        with an ideal (noiseless) simulator, and once with the given
        `noise_model` - and compares both sampled distributions against
        the exact theoretical distribution.

        Returns:
          - tvd_sampling_only: TVD between theoretical and the
            noiseless/ideal sampled distribution (sampling noise alone).
          - tvd_sampling_and_noise: TVD between theoretical and the
            noisy sampled distribution (sampling noise + hardware noise).
          - observed_noise_delta: tvd_sampling_and_noise -
            tvd_sampling_only. This is the *observed, empirical* change
            in TVD between the two experiments, for this circuit and
            this noise model - not a decomposition of "how much error
            noise contributes" in any exact/analytical sense, since
            sampling noise and hardware noise aren't independent,
            additive quantities.
        """
        theoretical = self.get_probabilities()

        ideal_counts = self.run_measurement(shots=shots, seed_simulator=seed_simulator)
        ideal_measured = counts_to_probabilities(ideal_counts)
        tvd_sampling_only = compare_distributions(theoretical, ideal_measured)[
            "total_variation_distance"
        ]

        noisy_counts = self.run_measurement(
            shots=shots, noise_model=noise_model, seed_simulator=seed_simulator
        )
        noisy_measured = counts_to_probabilities(noisy_counts)
        tvd_sampling_and_noise = compare_distributions(theoretical, noisy_measured)[
            "total_variation_distance"
        ]

        return {
            "theoretical": theoretical,
            "ideal_measured": ideal_measured,
            "noisy_measured": noisy_measured,
            "tvd_sampling_only": tvd_sampling_only,
            "tvd_sampling_and_noise": tvd_sampling_and_noise,
            "observed_noise_delta": tvd_sampling_and_noise - tvd_sampling_only,
        }

    # =====================================================
    # Display helpers (printing only - no computation of their own)
    # =====================================================
    def show_circuit(self) -> None:
        """Prints a text (ASCII) drawing of the circuit"""
        print("\n--- Circuit Diagram ---")
        print(self.qc.draw(output="text"))

    def show_statevector(self) -> None:
        state = self.get_statevector()
        print("\n--- Final Statevector ---")
        print(state)
        print("\n--- Theoretical Probabilities ---")
        for outcome, p in sorted(state.probabilities_dict().items()):
            print(f"|{outcome}>: {p:.4f}")

    def show_measurement(
        self,
        shots: int = 1024,
        qubits: list[int] | None = None,
        noise_model: NoiseModel | None = None,
        seed_simulator: int | None = None,
        plot: bool = True,
    ) -> dict[str, int]:
        """Runs `run_measurement` and prints/plots the result. Returns the counts."""
        counts = self.run_measurement(
            shots=shots, qubits=qubits, noise_model=noise_model, seed_simulator=seed_simulator
        )

        qubits_label = f", qubits={qubits}" if qubits is not None else ", all qubits"
        noise_label = ", with noise model" if noise_model is not None else ""
        print(f"\n--- Measurement Results ({shots} shots{qubits_label}{noise_label}) ---")
        for outcome, count in sorted(counts.items()):
            print(f"{outcome}: {count} ({count/shots*100:.1f}%)")

        if plot:
            filename = f"{self.qc.name}_histogram.png"
            save_histogram(counts, filename)
            print(f"\n[Saved histogram plot -> {filename}]")

        return counts

    def show_comparison(
        self, shots: int = 4096, seed_simulator: int | None = None
    ) -> dict[str, object]:
        """Runs `compare_theoretical_vs_measured` and prints a readable table. Returns the result dict."""
        result = self.compare_theoretical_vs_measured(shots=shots, seed_simulator=seed_simulator)
        theoretical = result["theoretical"]
        measured = result["measured"]
        all_outcomes = sorted(set(theoretical) | set(measured))

        print(f"\n--- Theoretical vs Measured ({shots} shots) ---")
        print(f"{'Outcome':<10}{'Theoretical':<14}{'Measured':<12}{'Abs. Error':<12}")
        for outcome in all_outcomes:
            t = theoretical.get(outcome, 0.0)
            m = measured.get(outcome, 0.0)
            print(f"|{outcome}>{'':<3}{t:<14.4f}{m:<12.4f}{abs(t - m):<12.4f}")

        print(f"\nTotal absolute error : {result['total_absolute_error']:.4f}")
        print(f"Mean absolute error   : {result['mean_absolute_error']:.4f}")
        print(f"Max absolute error    : {result['max_absolute_error']:.4f}")
        print(
            f"Total Variation Dist. : {result['total_variation_distance']:.4f}  "
            "(0 = identical, 1 = fully different)"
        )
        print(
            "(This quantifies the statistical difference between exact "
            "probabilities and sampled measurement results - small values "
            "here are expected and shrink as the number of shots grows; "
            "this is normal statistical noise from the measurement "
            "process, not a measure of simulator accuracy.)"
        )
        return result

    def summary(self) -> None:
        print(f"\n=== Circuit: {self.qc.name} ({self.num_qubits} qubits) ===")
        print("Gates applied in order:")
        for i, g in enumerate(self.history, 1):
            print(f"  {i}. {g}")
