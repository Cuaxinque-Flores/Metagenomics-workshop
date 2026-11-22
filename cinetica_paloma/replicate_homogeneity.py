#!/usr/bin/env python3
"""Replicate homogeneity per strain: individual wells, CV over time, half-plate comparison."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_kinetics import LAYOUT, parse

STRAINS = {"C": "Kosakonia", "D": "Phytobacter", "E": "Azospirillum", "F": "Paenibacillus", "G": "Pseudomonas"}


def main(path, outdir):
    out = Path(outdir)
    df = parse(path)
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
    cmap = plt.get_cmap("viridis")
    stats = []
    for k, (row, name) in enumerate(STRAINS.items()):
        cols = [f"{row}{i}" for i in range(1, 13)]
        sub = df[cols]
        ax = axes[0, k]
        for i, c in enumerate(cols):
            ax.plot(sub.index, sub[c], color=cmap(i / 11), lw=1, alpha=0.9)
        ax.plot(sub.index, sub.mean(axis=1), color="black", lw=2)
        ax.set_title(name)
        if k == 0:
            ax.set_ylabel("OD630")
        cv = sub.std(axis=1) / sub.mean(axis=1) * 100
        axes[1, k].plot(sub.index, cv, color="firebrick")
        axes[1, k].set_xlabel("Time (h)")
        if k == 0:
            axes[1, k].set_ylabel("CV (%)")
        axes[1, k].set_ylim(0, 40)
        h1, h2 = sub.iloc[:, :6].mean(axis=1), sub.iloc[:, 6:].mean(axis=1)
        late = sub.index >= 6
        stats.append({
            "strain": name,
            "CV_mean_all_%": cv.mean(), "CV_mean_after_6h_%": cv[late].mean(),
            "CV_at_24h_%": cv.iloc[-1],
            "cols1-6_vs_7-12_mean_abs_diff_OD": (h1 - h2).abs().mean(),
            "min_pairwise_pearson_r": sub.corr().values[np.triu_indices(12, 1)].min(),
        })
    fig.suptitle("Replicate wells per strain (12 wells each; black = mean)")
    fig.tight_layout()
    fig.savefig(out / "replicates_homogeneity.png", dpi=200)
    pd.DataFrame(stats).to_csv(out / "replicate_homogeneity.csv", index=False, float_format="%.3f")
    print(pd.DataFrame(stats).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
