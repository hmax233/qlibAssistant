#!/usr/bin/env python3
"""Plot grouped strict net returns from a strict_summary.csv report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule", default="baseline")
    args = parser.parse_args()
    frame = pd.read_csv(args.summary)
    frame = frame[frame["rule"].eq(args.rule)].copy()
    sources = frame["source"].drop_duplicates().tolist()
    topks = sorted(frame["topk"].unique())
    x = np.arange(len(topks), dtype=float)
    width = 0.8 / max(len(sources), 1)
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    for index, source in enumerate(sources):
        selected = frame[frame["source"].eq(source)].set_index("topk").reindex(topks)
        values = selected["net_cumulative"].to_numpy(dtype=float) * 100
        offset = (index - (len(sources) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=source)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
    benchmark = float(frame["CSI1000_cumulative"].dropna().iloc[0]) * 100
    ax.axhline(benchmark, color="black", linestyle="--", linewidth=1.3,
               label=f"CSI1000 {benchmark:.1f}%")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x, [f"Top{k}" for k in topks])
    ax.set_ylabel("Strict net cumulative return (%)")
    ax.set_title(f"Common-universe strict comparison ({args.rule})")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2, loc="upper right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
