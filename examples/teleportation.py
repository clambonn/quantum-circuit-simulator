"""
Quantum Teleportation Example
================================
Transfers a quantum state from q0 to q2 using a shared Bell pair + two
classical measurements, without transferring the physical qubit itself.

Design note: teleportation needs mid-circuit measurement (measuring a
qubit partway through the circuit and using the result immediately) and
classical control (executing a gate only if the measurement result was
1). Neither of these exists in the simple CircuitSimulator in
src/quantum_circuit_simulator/simulator.py, so this example uses Qiskit
directly instead of the class, with a full physical explanation at
every step.

The physical idea:
  1. We have a qubit called q0 holding a quantum state we want to
     transfer ("send") from Alice to Bob, without physically sending the
     qubit itself.
  2. Alice and Bob start out sharing an entangled Bell pair (q1 with
     Alice, q2 with Bob) — this is the "quantum channel" between them.
  3. Alice performs operations on q0 and q1, measures them, and sends the
     result (ordinary classical bits — could be a phone call or an
     email) to Bob.
  4. Bob uses that result to know what simple correction (X and/or Z) to
     apply to q2, after which q2 holds exactly the same state that was
     originally in q0. Important distinction: the physical qubit (q0)
     itself doesn't go anywhere — it still physically exists. What
     happens is that the quantum state (the state that used to be in it)
     is destroyed by the measurement in Step 3, because measurement
     forces the superposition to collapse into a definite classical
     state. That's exactly why we can say "the state was teleported"
     rather than "copied" — consistent with the no-cloning theorem
     (there's never a moment where two copies of the same quantum state
     exist at the same time).

State preparation: q0 is prepared with Ry(theta) followed by Rz(phi),
producing the general single-qubit state

    |psi> = cos(theta/2)|0> + e^(i*phi) * sin(theta/2)|1>

rather than only the "real-amplitude" states Ry(theta) alone can reach.
Using both theta and phi means the test covers phase, not just
probabilities — a state with an incorrect phase can still measure the
right *probabilities* on the computational basis, so probability-only
verification isn't enough to prove teleportation is fully correct.
"""

import math

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator


def quantum_teleportation_demo(theta: float = 0.9, phi: float = 0.0, verbose: bool = True):
    """
    Builds and runs a complete quantum teleportation circuit, and verifies
    that the state successfully reached Bob, for the specific state
    prepared with Ry(theta) followed by Rz(phi).

    Accuracy note: the result here proves teleportation succeeded for
    *this specific state*, not a general mathematical proof for every
    possible state. For a more general proof, run
    teleportation_multi_state_test(), which tries several (theta, phi) pairs.

    theta, phi: used to prepare q0 in a general state (not a simple |0>
    or |1>, and not restricted to real amplitudes) via Ry(theta) then
    Rz(phi), so we can demonstrate that teleportation preserves both
    probability *and* relative phase, not just measurement probabilities.
    """
    qr = QuantumRegister(3, "q")   # q0=message, q1=Alice's half, q2=Bob's half
    cr = ClassicalRegister(2, "c")  # c[0] <- measure(q0), c[1] <- measure(q1)
    verify_cr = ClassicalRegister(1, "verify")  # only for the final verification measurement

    qc = QuantumCircuit(qr, cr, verify_cr, name="teleportation")

    # ---------- Step 0: prepare the state to be transferred ----------
    # Put q0 into a general state (not |0> or |1>, and with a nontrivial
    # relative phase) using Ry then Rz, so we make sure teleportation
    # works for any state, not just the easy real-amplitude cases
    qc.ry(theta, qr[0])
    qc.rz(phi, qr[0])
    qc.barrier()

    # ---------- Step 1: prepare the Bell pair shared between Alice and Bob ----------
    qc.h(qr[1])
    qc.cx(qr[1], qr[2])
    qc.barrier()

    # ---------- Step 2: Alice entangles the message (q0) with her half of the pair (q1) ----------
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.barrier()

    # ---------- Step 3: Alice measures q0 and q1 (measurement destroys their original state) ----------
    qc.measure(qr[0], cr[0])
    qc.measure(qr[1], cr[1])
    qc.barrier()

    # ---------- Step 4: Bob corrects q2 based on Alice's measurement results ----------
    # if c1 (the measurement result of q1) = 1 -> apply X to q2
    with qc.if_test((cr[1], 1)):
        qc.x(qr[2])
    # if c0 (the measurement result of q0) = 1 -> apply Z to q2
    with qc.if_test((cr[0], 1)):
        qc.z(qr[2])
    qc.barrier()

    # ---------- Step 5: verification ----------
    # If teleportation succeeded, q2 now holds exactly the same state q0
    # originally had (Rz(phi)Ry(theta)|0>). We apply the inverse
    # rotations (Rz(-phi) then Ry(-theta)) to "rotate it back" to |0>,
    # and if the measurement comes out 0 with a rate close to 100%,
    # that's strong evidence the transfer succeeded for this specific
    # state (not a general proof for every possible state)
    qc.rz(-phi, qr[2])
    qc.ry(-theta, qr[2])
    qc.measure(qr[2], verify_cr[0])

    if verbose:
        print("\n--- Teleportation Circuit ---")
        print(qc.draw(output="text"))

    simulator = AerSimulator()
    shots = 4096
    job = simulator.run(qc, shots=shots)
    counts = job.result().get_counts()

    # Tally only the verification bit results (the last classical
    # register), regardless of Alice's measurement outcomes, to see
    # whether the transfer always succeeds
    verify_zero = 0
    verify_one = 0
    for outcome, count in counts.items():
        # the outcome string looks like "verify_bit alice_bits" (Qiskit orders them in reverse)
        verify_bit = outcome.split()[0]
        if verify_bit == "0":
            verify_zero += count
        else:
            verify_one += count

    success_rate = verify_zero / shots * 100

    if verbose:
        print(f"\n--- Verification Results ({shots} shots) ---")
        print(f"Success (verify qubit = 0): {verify_zero} ({success_rate:.2f}%)")
        print(f"Failure (verify qubit = 1): {verify_one} ({100 - success_rate:.2f}%)")
        print("\nIf the first percentage is close to 100%, this is strong")
        print("evidence that q0's original state was successfully")
        print("teleported to q2 for this (theta, phi), including its")
        print("relative phase, even though the original quantum state was")
        print("destroyed by the measurement (the physical qubit itself")
        print("still exists) and no quantum information was physically")
        print("sent other than the Bell pair qubits + 2 ordinary classical bits.")

    return {"counts": counts, "success_rate": success_rate, "theta": theta, "phi": phi}


def teleportation_multi_state_test(states: list[tuple[float, float]] | None = None):
    """
    Runs quantum_teleportation_demo across several different (theta, phi)
    pairs, to provide more general evidence (not just for a single case)
    that the circuit works correctly for any general state - including
    relative phase, not just the computational-basis probabilities.

    `states`: a list of (theta, phi) tuples to test. If not given,
    defaults to a small mix of theta values with phi=0.0 (i.e.
    real-amplitude states only).
    """
    if states is None:
        states = [(theta, 0.0) for theta in (0, math.pi / 4, math.pi / 2, math.pi, 2.35)]

    print("\n" + "=" * 50)
    print("Teleportation across multiple states (theta, phi pairs)")
    print("=" * 50)

    results = []
    for theta, phi in states:
        r = quantum_teleportation_demo(theta=theta, phi=phi, verbose=False)
        results.append(r)
        print(
            f"theta = {theta:.4f} rad, phi = {phi:.4f} rad  ->  "
            f"success rate = {r['success_rate']:.2f}%"
        )

    all_passed = all(r["success_rate"] > 99.0 for r in results)
    print(f"\nAll tested states teleported successfully: {all_passed}")
    return results


def main():
    quantum_teleportation_demo(theta=0.9, phi=0.0)
    teleportation_multi_state_test(
        states=[
            (0.0, 0.0),
            (math.pi / 4, 0.0),
            (math.pi / 2, math.pi / 2),
            (math.pi / 3, math.pi),
        ]
    )


if __name__ == "__main__":
    main()
