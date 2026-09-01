"""
Visualization
==============
All matplotlib / plotting code lives here, so that `simulator.py`
doesn't need to know anything about files, figures, or how a
histogram should look. CircuitSimulator only ever returns data
(counts, probabilities, dicts); turning that data into a saved image
is a separate concern handled by this module.

Author: Yasmin Kasem
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram


def save_histogram(
    counts: dict[str, int],
    filename: str,
    title: str | None = None,
) -> str:
    """
    Renders a histogram of measurement counts and saves it to
    `filename`. Returns the filename for convenience.
    """
    figure = plot_histogram(counts)
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(filename, bbox_inches="tight")
    plt.close(figure)
    return filename


def save_benchmark_plot(
    qubit_counts: Sequence[int],
    times_seconds: Sequence[float],
    filename: str,
    title: str = "Statevector simulation time vs. number of qubits",
) -> str:
    """
    Plots execution time (seconds, log scale) against number of
    qubits (linear scale), which is the clearest way to show that
    statevector simulation cost grows exponentially (2^n amplitudes)
    with the number of qubits - a straight line on a log-y plot is the
    visual signature of exponential growth.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(qubit_counts, times_seconds, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Execution time (seconds, log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_noise_sweep_plot(
    probabilities: Sequence[float],
    tvd_values: Sequence[float],
    filename: str,
    title: str = "Readout error probability vs. Total Variation Distance",
    color: str | None = None,
) -> str:
    """
    Plots a swept noise probability (x-axis) against the resulting
    TVD between the noisy and ideal distributions (y-axis), to show
    how measured results degrade as noise increases.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_kwargs = {"marker": "o"}
    if color is not None:
        plot_kwargs["color"] = color
    ax.plot(probabilities, tvd_values, **plot_kwargs)
    ax.set_xlabel("Readout error probability")
    ax.set_ylabel("Total Variation Distance (noisy vs. ideal)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    return filename
