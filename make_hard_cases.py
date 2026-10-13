"""
make_hard_cases.py - rebuild results/figures/hard_cases.png from saved predictions.
Shows the 12 CORRECT test predictions closest to the decision threshold.
Reads results/test_predictions_<run>.csv only (no model, no torch needed).

    python make_hard_cases.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r18_basic")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--data-root", default="data/raw/Dataset")
    ap.add_argument("--results", default="results")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(Path(args.results) / f"test_predictions_{args.run}.csv")
    df["pred"] = (df["prob_fire"] >= args.thr).astype(int)
    ok = df[df["pred"] == df["label"]].copy()
    ok["margin"] = (ok["prob_fire"] - args.thr).abs()
    hc = ok.sort_values("margin").head(args.n)

    cols = 4
    rows = int(np.ceil(len(hc) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, hc.iterrows()):
        with Image.open(Path(args.data_root) / r["path"]) as im:
            ax.imshow(im.convert("RGB"))
        true = "fire" if r["label"] == 1 else "no fire"
        ax.set_title(f'{Path(r["path"]).name}\ntrue: {true}, P(fire)={r["prob_fire"]:.2f}', fontsize=8)
    fig.suptitle("Hard cases: correct predictions closest to the decision threshold")
    fig.tight_layout()
    out = Path(args.results) / "figures" / "hard_cases.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
