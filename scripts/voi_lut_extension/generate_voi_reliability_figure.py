#!/usr/bin/env python3
"""Generate reliability curves and bin-level data from cross-fitted OOF scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCORES = {
    "crossview_focal075": "Focal CrossView",
    "convex_stack": "Convex blend",
    "logistic_stack": "Logistic stack",
    "convex_stack_platt": "Platt calibrated",
    "convex_stack_isotonic": "Isotonic calibrated",
}


def reliability_bins(frame: pd.DataFrame, score: str, bins: int) -> pd.DataFrame:
    values = frame[["label", score]].dropna().copy()
    values["bin"] = pd.qcut(values[score], q=bins, labels=False, duplicates="drop")
    grouped = values.groupby("bin", observed=True)
    result = grouped.agg(
        n=("label", "size"),
        mean_predicted=(score, "mean"),
        observed_rate=("label", "mean"),
        positives=("label", "sum"),
    ).reset_index()

    # Wilson intervals remain finite for bins with no positive observations.
    z = 1.959963984540054
    n = result["n"].to_numpy(dtype=float)
    p = result["observed_rate"].to_numpy(dtype=float)
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denominator
    result["observed_ci_low"] = np.maximum(0.0, center - half_width)
    result["observed_ci_high"] = np.minimum(1.0, center + half_width)
    result.insert(0, "score", score)
    result.insert(1, "label_name", SCORES[score])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    missing = {"label", *SCORES} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    combined = pd.concat(
        [reliability_bins(frame, score, args.bins) for score in SCORES],
        ignore_index=True,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.5,
            label="Perfect calibration")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for (score, label), color in zip(SCORES.items(), colors):
        rows = combined[combined["score"] == score]
        yerr = np.vstack(
            [
                rows["observed_rate"] - rows["observed_ci_low"],
                rows["observed_ci_high"] - rows["observed_rate"],
            ]
        )
        ax.errorbar(
            rows["mean_predicted"],
            rows["observed_rate"],
            yerr=yerr,
            marker="o",
            markersize=4,
            linewidth=1.6,
            capsize=2,
            color=color,
            label=label,
        )

    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
           ylabel="Observed positive rate", title="Cross-fitted OOF reliability")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
