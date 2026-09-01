"""
Analysis
=========
Pure, side-effect-free functions for comparing probability
distributions (no printing, no plotting, no dependency on
CircuitSimulator or Qiskit). Kept separate from simulator.py so the
statistical methodology can be tested, reused, and reasoned about on
its own.

Author: Yasmin Kasem
"""



def counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    """
    Converts raw measurement counts (e.g. {"00": 512, "11": 488}) into
    a probability distribution (e.g. {"00": 0.5, "11": 0.476...}) by
    dividing every count by the total number of shots.
    """
    if not counts:
        raise ValueError("counts must not be empty")

    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts.values()
    ):
        raise ValueError("counts values must be non-negative integers")

    total = sum(counts.values())
    if total == 0:
        raise ValueError("counts must not be empty (total shots is 0)")

    return {outcome: count / total for outcome, count in counts.items()}


def total_variation_distance(
    distribution_a: dict[str, float], distribution_b: dict[str, float]
) -> float:
    """
    Total Variation Distance (TVD) between two probability
    distributions: 0.5 * sum(|p - q|) across every outcome that
    appears in either distribution.

    This is the standard metric for comparing two probability
    distributions over the same outcome space. It ranges from 0
    (identical distributions) to 1 (completely disjoint support), and
    is independent of the number of outcomes - which makes it more
    meaningful than a raw summed error when comparing across circuits
    with different numbers of qubits/outcomes.
    """
    outcomes = set(distribution_a) | set(distribution_b)
    return 0.5 * sum(
        abs(distribution_a.get(outcome, 0.0) - distribution_b.get(outcome, 0.0))
        for outcome in outcomes
    )


def mean_absolute_error(
    distribution_a: dict[str, float], distribution_b: dict[str, float]
) -> float:
    """Average |p - q| across every outcome that appears in either distribution."""
    outcomes = set(distribution_a) | set(distribution_b)
    if not outcomes:
        return 0.0
    total = sum(
        abs(distribution_a.get(outcome, 0.0) - distribution_b.get(outcome, 0.0))
        for outcome in outcomes
    )
    return total / len(outcomes)


def max_absolute_error(
    distribution_a: dict[str, float], distribution_b: dict[str, float]
) -> float:
    """The single largest |p - q| across any one outcome."""
    outcomes = set(distribution_a) | set(distribution_b)
    if not outcomes:
        return 0.0
    return max(
        abs(distribution_a.get(outcome, 0.0) - distribution_b.get(outcome, 0.0))
        for outcome in outcomes
    )


def compare_distributions(
    distribution_a: dict[str, float], distribution_b: dict[str, float]
) -> dict[str, object]:
    """
    Computes every distance metric between two probability
    distributions at once (total absolute error, MAE, max absolute
    error, and TVD), and returns both input distributions alongside
    them. This does not claim anything about "accuracy of simulation" -
    it purely quantifies how different two distributions are,
    regardless of what produced them (theoretical vs. sampled, ideal
    vs. noisy, or any other pair of distributions over the same
    outcome space).
    """
    outcomes = set(distribution_a) | set(distribution_b)
    errors = [
        abs(distribution_a.get(outcome, 0.0) - distribution_b.get(outcome, 0.0))
        for outcome in outcomes
    ]
    total_error = sum(errors)
    mean_error = total_error / len(errors) if errors else 0.0
    max_error = max(errors) if errors else 0.0
    tvd = 0.5 * total_error

    return {
        "distribution_a": distribution_a,
        "distribution_b": distribution_b,
        "total_absolute_error": total_error,
        "mean_absolute_error": mean_error,
        "max_absolute_error": max_error,
        "total_variation_distance": tvd,
    }
