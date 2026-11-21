#!/usr/bin/env python3
"""Kinetic analysis of the SynCom 5-strain plate (ELx808, OD630, 20-min interval, 24 h)."""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAYOUT = {
    "A": "Control", "B": "Control", "C": "Kosakonia", "D": "Phytobacter",
    "E": "Azospirillum", "F": "Paenibacillus", "G": "Pseudomonas", "H": "Control",
}


def parse(path):
    lines = Path(path).read_text(encoding="latin1").splitlines()
    start = next(i for i, l in enumerate(lines) if re.match(r"Time\tT.*Read 2", l))
    header = lines[start].split("\t")
    recs = [l.split("\t") for l in lines[start + 1:] if re.match(r"\d+:\d\d:\d\d\t", l)]
    df = pd.DataFrame(recs, columns=header[: len(recs[0])]).drop(columns=lambda c: c.startswith("T"), errors="ignore")
    df = df.iloc[:, [0] + [i for i, c in enumerate(df.columns) if re.fullmatch(r"[A-H]\d+", c)]]
    h, m, s = zip(*[map(int, t.split(":")) for t in df["Time"]])
    df.insert(0, "hours", np.array(h) + np.array(m) / 60 + np.array(s) / 3600)
    df = df.drop(columns="Time").set_index("hours").astype(float)
    return df


def growth_params(y, window=6, floor=0.02):
    """Max specific growth rate from the steepest log-linear window; lag by tangent at that window."""
    t = y.index.values
    v = y.values
    best = (np.nan, np.nan, np.nan, np.nan)
    for i in range(len(t) - window):
        yy = v[i:i + window]
        if (yy <= floor).any():
            continue
        slope, icpt = np.polyfit(t[i:i + window], np.log(yy), 1)
        if np.isnan(best[0]) or slope > best[0]:
            best = (slope, icpt, i, 0)
    slope, icpt = best[0], best[1]
    lag = (np.log(max(v[0], floor)) - icpt) / slope if slope == slope else np.nan
    return slope, max(lag, 0.0), np.nanmax(v)


def main(path, outdir):
    out = Path(outdir)
    out.mkdir(exist_ok=True)
    df = parse(path)
    df.to_csv(out / "od630_raw_wells.csv")

    blank = df[[c for c in df if LAYOUT[c[0]] == "Control"]].mean(axis=1)
    corr = df.sub(blank, axis=0)
    rows = []
    for row, name in LAYOUT.items():
        cols = [c for c in df if c[0] == row]
        sub = df[cols]
        rows.append({
            "row": row, "group": name, "n_wells": len(cols),
            "OD_t0": sub.iloc[0].mean(), "OD_max": sub.max().mean(),
            "OD_final": sub.iloc[-1].mean(), "delta_final_t0": (sub.iloc[-1] - sub.iloc[0]).mean(),
            "delta_max_t0": (sub.max() - sub.iloc[0]).mean(),
            "t_at_max_h": sub.mean(axis=1).idxmax(),
        })
    pos = {w: growth_params(corr[w]) for w in corr if LAYOUT[w[0]] != "Control"}
    gp = pd.DataFrame(pos, index=["mu_max_per_h", "lag_h", "OD_corr_max"]).T
    gp["row"] = [w[0] for w in gp.index]
    gp["group"] = gp["row"].map(LAYOUT)
    gp.to_csv(out / "growth_parameters_by_well.csv", float_format="%.4f")
    agg = gp.groupby(["row", "group"]).agg(
        mu_max_mean=("mu_max_per_h", "mean"), mu_max_sd=("mu_max_per_h", "std"),
        lag_mean_h=("lag_h", "mean"), lag_sd_h=("lag_h", "std"),
        ODmax_corr_mean=("OD_corr_max", "mean"), ODmax_corr_sd=("OD_corr_max", "std")).reset_index()
    agg["doubling_time_h"] = np.log(2) / agg["mu_max_mean"]
    agg.to_csv(out / "growth_parameters_summary.csv", index=False, float_format="%.4f")
    print(agg.to_string(index=False, float_format="%.3f"))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary_by_row.csv", index=False, float_format="%.4f")
    print(summ.to_string(index=False, float_format="%.3f"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("tab10")
    for k, (name, rws) in enumerate({"Control": "ABH", "Kosakonia": "C", "Phytobacter": "D",
                                     "Azospirillum": "E", "Paenibacillus": "F", "Pseudomonas": "G"}.items()):
        cols = [c for c in df if c[0] in rws]
        m, sd = df[cols].mean(axis=1), df[cols].std(axis=1)
        ax.plot(df.index, m, label=name, color="black" if name == "Control" else cmap(k), lw=1.8)
        ax.fill_between(df.index, m - sd, m + sd, color="grey" if name == "Control" else cmap(k), alpha=0.2)
    ax.set(xlabel="Time (h)", ylabel="OD630", title="SynCom 5 strains, 30 C, 24 h")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "kinetics_od630.png", dpi=200)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
