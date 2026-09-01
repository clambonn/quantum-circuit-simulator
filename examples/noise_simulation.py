"""
Noise Simulation Example
==========================
Demonstrates the difference between:

    Ideal (theoretical) distribution
              vs.
    Noisy simulated distribution
              vs.
    (Noiseless) measured distribution

using a Bell state, and then sweeps the readout error probability to
show how measured results degrade as noise increases.

This is what turns the project from "quantum computing under ideal
conditions" into "quantum simulation that also models the imperfect,
noisy hardware quantum computers actually run on".
"""

from quantum_circuit_simulator import CircuitSimulator, NoiseConfig, build_noise_model
from quantum_circuit_simulator.analysis import (
    compare_distributions,
    counts_to_probabilities,
)
from quantum_circuit_simulator.visualization import save_noise_sweep_plot


def compare_ideal_noisy_measured():
    """
    Builds a Bell state and compares three distributions at once:
    theoretical (exact), noiseless-measured (sampling noise only), and
    noisy-measured (sampling noise + simulated hardware noise).
    """
    bell = CircuitSimulator(num_qubits=2, name="bell_state_noise_demo")
    bell.h(0).cx(0, 1)

    noise_config = NoiseConfig(
        bit_flip=0.01,
        phase_flip=0.01,
        depolarizing=0.02,
        readout=0.05,
    )
    noise_model = build_noise_model(noise_config)

    result = bell.compare_noise_impact(noise_model, shots=8192, seed_simulator=7)

    print("--- Ideal vs. Noisy vs. Measured (Bell state, 8192 shots) ---")
    print("Theoretical (exact):        ", result["theoretical"])
    print("Measured, no noise model:   ", result["ideal_measured"])
    print("Measured, with noise model: ", result["noisy_measured"])
    print(f"\nTVD (theoretical vs. noiseless measured) : {result['tvd_sampling_only']:.4f}")
    print(f"TVD (theoretical vs. noisy measured)      : {result['tvd_sampling_and_noise']:.4f}")
    print(f"Observed noise delta (noisy TVD - noiseless TVD): {result['observed_noise_delta']:.4f}")
    print(
        "\n(This is the *observed, empirical* change between the two "
        "experiments for this circuit and this noise model - not a "
        "precise decomposition of how much error 'belongs to' noise "
        "versus sampling, since they aren't independent, additive "
        "quantities.)"
    )
    return result


def readout_error_sweep(probabilities=None, shots=4096):
    """
    Sweeps the readout error probability from low to high and measures
    the resulting Total Variation Distance against the ideal (noiseless)
    theoretical distribution, to show that TVD increases roughly in
    step with the injected error rate.
    """
    if probabilities is None:
        probabilities = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]

    bell = CircuitSimulator(num_qubits=2, name="bell_state_readout_sweep")
    bell.h(0).cx(0, 1)
    theoretical = bell.get_probabilities()

    print("\n--- Readout Error Sweep (Bell state) ---")
    print(f"{'Readout p':<12}{'TVD':<10}")

    tvd_values = []
    for p in probabilities:
        noise_model = build_noise_model(NoiseConfig(readout=p))
        counts = bell.run_measurement(shots=shots, noise_model=noise_model, seed_simulator=123)
        measured = counts_to_probabilities(counts)
        tvd = compare_distributions(theoretical, measured)["total_variation_distance"]
        tvd_values.append(tvd)
        print(f"{p:<12.2f}{tvd:<10.4f}")

    filename = save_noise_sweep_plot(probabilities, tvd_values, "readout_error_sweep.png")
    print(f"\n[Saved sweep plot -> {filename}]")
    return probabilities, tvd_values


def main():
    compare_ideal_noisy_measured()
    readout_error_sweep()


if __name__ == "__main__":
    main()
