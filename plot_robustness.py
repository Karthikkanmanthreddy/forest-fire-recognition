"""
plot_robustness.py - redraw results/figures/robustness.png from results/robustness.csv.
Reads saved numbers only; does not load any model or image.

    python plot_robustness.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEAN = {"brightness factor": 1.0, "gaussian blur radius": 0.0,
         "resolution scale": 1.0, "JPEG quality": 100.0}
ORDER = ["brightness factor", "gaussian blur radius", "resolution scale", "JPEG quality"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/robustness.csv")
    ap.add_argument("--out", default="results/figures/robustness.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    models = list(dict.fromkeys(df["model"]))
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharey="row")
    for j, fam in enumerate(ORDER):
        sub = df[df.perturbation == fam]
        levels = sorted(sub["level"].unique())
        pos = {lv: i for i, lv in enumerate(levels)}
        for i, (metric, lab) in enumerate((("fire_recall", "fire recall"),
                                           ("fire_precision", "fire precision"))):
            ax = axes[i, j]
            for m in models:
                s = sub[sub.model == m].sort_values("level")
                ax.plot([pos[v] for v in s["level"]], s[metric], marker="o", label=m)
            ax.axvline(pos[CLEAN[fam]], ls="--", c="gray", lw=1)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels([f"{v:g}" for v in levels], fontsize=8)
            ax.set_ylim(0.4, 1.02)
            ax.set_title(fam, fontsize=10)
            if j == 0:
                ax.set_ylabel(lab)
            ax.set_xlabel("level (dashed line = unperturbed)", fontsize=8)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
