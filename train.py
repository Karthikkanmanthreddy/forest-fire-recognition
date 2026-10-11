"""
train.py - train one model; select epoch and decision threshold on VALIDATION only.
The test set is never loaded here.

Examples
    python train.py --arch cnn --epochs 30 --name cnn_baseline
    python train.py --arch mobilenet_v3_small --epochs 15 --freeze-epochs 3 --name mnv3_basic
    python train.py --arch resnet18 --epochs 15 --aug strong --name r18_strong
    python train.py --arch cnn --limit 64 --epochs 1 --name smoke      # quick check
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fire_common import (FireDataset, make_transforms, build_model, head_param_ids,
                         predict_probs, binary_metrics, choose_threshold,
                         set_seed, get_device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["cnn", "mobilenet_v3_small", "resnet18"])
    ap.add_argument("--name", required=True, help="run name; outputs go to runs/<name>")
    ap.add_argument("--data-root", default="data/raw/Dataset")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--freeze-epochs", type=int, default=0,
                    help="transfer models: epochs with the backbone frozen (head only)")
    ap.add_argument("--lr", type=float, default=1e-3, help="lr (cnn / frozen head phase)")
    ap.add_argument("--lr-ft", type=float, default=1e-4, help="lr after unfreezing")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--aug", default="basic", choices=["none", "basic", "strong"])
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: use N images")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    pretrained = args.arch != "cnn"
    print(f"device={device}  arch={args.arch}  pretrained={pretrained}")

    sp = Path(args.splits)
    ds_tr = FireDataset(sp / "train.csv", args.data_root,
                        make_transforms(True, args.aug), args.limit)
    ds_va = FireDataset(sp / "val.csv", args.data_root,
                        make_transforms(False), args.limit)
    dl_tr = DataLoader(ds_tr, args.batch_size, shuffle=True, num_workers=args.workers,
                       drop_last=len(ds_tr) > args.batch_size)
    dl_va = DataLoader(ds_va, args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"train={len(ds_tr)}  val={len(ds_va)}")

    model, _ = build_model(args.arch, pretrained)
    model.to(device)
    head_ids = head_param_ids(model, args.arch)
    n_params = sum(p.numel() for p in model.parameters())
    crit = nn.BCEWithLogitsLoss()

    def make_opt(frozen: bool):
        if frozen:
            for p in model.parameters():
                p.requires_grad = id(p) in head_ids
            params, lr = [p for p in model.parameters() if p.requires_grad], args.lr
        else:
            for p in model.parameters():
                p.requires_grad = True
            params = list(model.parameters())
            lr = args.lr if not pretrained else args.lr_ft
        return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)

    frozen = pretrained and args.freeze_epochs > 0
    opt = make_opt(frozen)

    hist, best_ap, best_ep, bad = [], -1.0, -1, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        if frozen and ep > args.freeze_epochs:
            frozen = False
            opt = make_opt(False)
            print(f"-- epoch {ep}: backbone unfrozen, lr={args.lr_ft}")
        model.train()
        tl, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y); n += len(y)
        pv, yv = predict_probs(model, dl_va, device)
        vl = float(crit(torch.logit(torch.tensor(pv).clamp(1e-6, 1 - 1e-6)),
                        torch.tensor(yv, dtype=torch.float32)))
        m = binary_metrics(yv, pv, 0.5)
        ap_val = average_precision_score(yv, pv)
        hist.append({"epoch": ep, "train_loss": tl / n, "val_loss": vl,
                     "val_acc": m["accuracy"], "val_f1": m["fire_f1"],
                     "val_recall": m["fire_recall"], "val_ap": ap_val,
                     "phase": "head" if frozen else "full"})
        print(f"ep {ep:02d} | train_loss {tl/n:.4f} | val_loss {vl:.4f} | "
              f"val_acc {m['accuracy']:.3f} | val_f1 {m['fire_f1']:.3f} | val_AP {ap_val:.3f}")
        if ap_val > best_ap:                        # model selection: validation AP
            best_ap, best_ep, bad = ap_val, ep, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience and not frozen:
                print(f"early stop at epoch {ep}")
                break

    # reload the best epoch, fix the decision threshold on validation
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    pv, yv = predict_probs(model, dl_va, device)
    thr = choose_threshold(yv, pv, args.target_recall)
    val_at_05, val_at_thr = binary_metrics(yv, pv, 0.5), binary_metrics(yv, pv, thr)
    np.save(out / "val_probs.npy", pv)
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    cfg = vars(args) | {"best_epoch": best_ep, "best_val_ap": best_ap,
                        "val_threshold": thr, "n_params": n_params,
                        "train_minutes": (time.time() - t0) / 60, "device": str(device),
                        "val_metrics_at_0.5": val_at_05, "val_metrics_at_val_thr": val_at_thr}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.val_f1, label="val F1 (fire)"); ax[1].plot(h.epoch, h.val_ap, label="val AP")
    ax[1].axvline(best_ep, ls="--", c="gray"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle(args.name); fig.tight_layout(); fig.savefig(out / "curves.png", dpi=130)

    print(f"\nbest epoch {best_ep}  val_AP {best_ap:.4f}  threshold(val, recall>="
          f"{args.target_recall}) = {thr:.4f}")
    print("val @0.5     :", {k: round(v, 3) for k, v in val_at_05.items()})
    print("val @val-thr :", {k: round(v, 3) for k, v in val_at_thr.items()})
    print("saved to", out.resolve())


if __name__ == "__main__":
    main()
