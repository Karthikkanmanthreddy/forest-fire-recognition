"""
evaluate.py - FINAL test-set evaluation of runs produced by train.py.
Run this once, after all tuning is finished. Uses each run's decision
threshold that was fixed on the validation set.

Example
    python evaluate.py --runs cnn_baseline mnv3_basic --gallery-run mnv3_basic
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fire_common import (FireDataset, make_transforms, build_model, predict_probs,
                         binary_metrics, get_device, CLASS_NAMES)

METRICS = ["accuracy", "fire_precision", "fire_recall", "fire_f1", "ap"]


def bootstrap_ci(y, p, thr, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = {k: [] for k in METRICS}
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        m = binary_metrics(y[b], p[b], thr)
        for k in METRICS:
            vals[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in vals.items()}


def cm_plot(ax, m, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)


def gallery(df, root, kind, out_png, n=12, cols=4):
    sub = df[df["error"] == kind]
    if kind == "FP":
        sub = sub.sort_values("prob_fire", ascending=False)     # most confident errors
    else:
        sub = sub.sort_values("prob_fire", ascending=True)
    sub = sub.head(n)
    if sub.empty:
        return 0
    rows = int(np.ceil(len(sub) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.1 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, sub.iterrows()):
        with Image.open(Path(root) / r["path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f'{Path(r["path"]).name}\nP(fire)={r["prob_fire"]:.2f}', fontsize=8)
    fig.suptitle({"FP": "False positives (no fire predicted as fire)",
                  "FN": "False negatives (missed fire)"}[kind])
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names under runs/")
    ap.add_argument("--gallery-run", default=None, help="run used for the error gallery")
    ap.add_argument("--data-root", default="data/raw/Dataset")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    device = get_device()
    out = Path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    ds = FireDataset(Path(args.splits) / "test.csv", args.data_root, make_transforms(False))
    dl = DataLoader(ds, 32, shuffle=False, num_workers=args.workers)
    print(f"test images: {len(ds)}")

    rows, store = [], {}
    for name in args.runs:
        run = Path("runs") / name
        cfg = json.loads((run / "config.json").read_text())
        model, _ = build_model(cfg["arch"], pretrained=False)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        model.to(device)
        p, y = predict_probs(model, dl, device)
        thr = cfg["val_threshold"]
        store[name] = (p, y, thr, cfg)

        pred_df = ds.df[["path", "label"]].copy()
        pred_df["prob_fire"] = p
        pred_df["pred_at_val_thr"] = (p >= thr).astype(int)
        pred_df["error"] = np.where((pred_df.pred_at_val_thr == 1) & (pred_df.label == 0), "FP",
                             np.where((pred_df.pred_at_val_thr == 0) & (pred_df.label == 1), "FN", ""))
        pred_df.to_csv(out / f"test_predictions_{name}.csv", index=False)

        for tag, t in (("thr=0.5", 0.5), (f"thr={thr:.3f} (val, recall target)", thr)):
            m = binary_metrics(y, p, t)
            ci = bootstrap_ci(y, p, t, args.n_boot)
            row = {"model": name, "arch": cfg["arch"], "params": cfg["n_params"],
                   "operating_point": tag, "n_test": len(y)}
            for k in METRICS:
                row[k] = f"{m[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
            row |= {"TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"]}
            rows.append(row)
            print(name, tag, {k: round(m[k], 3) for k in METRICS})

    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    (out / "comparison.md").write_text(table.to_markdown(index=False)
                                       if hasattr(table, "to_markdown") else table.to_string())

    # confusion matrices at the validation-chosen threshold
    fig, axes = plt.subplots(1, len(store), figsize=(3.6 * len(store), 3.4))
    axes = np.atleast_1d(axes)
    for ax, (name, (p, y, thr, _)) in zip(axes, store.items()):
        cm_plot(ax, binary_metrics(y, p, thr), f"{name}\nthr={thr:.3f}")
    fig.tight_layout(); fig.savefig(out / "figures/confusion_matrices.png", dpi=130); plt.close(fig)

    # PR curves
    fig, ax = plt.subplots(figsize=(4.8, 4))
    for name, (p, y, thr, _) in store.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=name)
    ax.set_xlabel("recall (fire)"); ax.set_ylabel("precision (fire)")
    ax.set_ylim(0.4, 1.02); ax.legend(); ax.set_title("Test precision-recall")
    fig.tight_layout(); fig.savefig(out / "figures/pr_curves.png", dpi=130); plt.close(fig)

    # error gallery
    g = args.gallery_run or args.runs[-1]
    df = pd.read_csv(out / f"test_predictions_{g}.csv")
    nfp = gallery(df, args.data_root, "FP", out / "figures/false_positives.png")
    nfn = gallery(df, args.data_root, "FN", out / "figures/false_negatives.png")
    print(f"\nerror gallery for '{g}': {nfp} FP and {nfn} FN images shown")
    print("results written to", out.resolve())


if __name__ == "__main__":
    main()
