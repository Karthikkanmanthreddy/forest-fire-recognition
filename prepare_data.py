"""
prepare_data.py
Smoke and Fire Recognition in Forest Images - data preparation.

What this script does
1. Scans the Mendeley "Dataset for Forest Fire Detection" folder layout:
       <data-root>/Testing/{fire,nofire}
       <data-root>/Training and Validation/{fire,nofire}
2. Verifies every image opens and records size / mode.
3. Computes an exact hash (MD5) and a perceptual hash (pHash) per image.
4. Groups exact and near-duplicate images into clusters (union-find on
   Hamming distance <= --hash-thresh).
5. Resolves clusters so that no near-duplicate leaks across splits:
     - a cluster with conflicting labels (fire AND nofire)  -> all removed
     - a cluster that touches the authors' test folder       -> test copy kept
       (one per cluster), every training-folder copy removed
     - a cluster only inside the training folder             -> kept as one
       group; it is never split between train and val
6. Splits the remaining training-folder images into train / val with a
   stratified, group-aware split (fixed seed). The authors' Testing folder
   is used unchanged (apart from de-duplication) as the locked test set.
7. Writes CSVs, a removal log, a JSON summary and a duplicate-example figure.

Usage
    python prepare_data.py --data-root data/raw/Dataset --out data/splits

Dataset: Khan, A. and Hassan, B. (2020) "Dataset for Forest Fire Detection",
Mendeley Data, V1, doi:10.17632/gjmr63rz2r.1  (CC BY 4.0)
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
LABELS = {"nofire": 0, "fire": 1}


# ----------------------------------------------------------------------------
# 1. scan
# ----------------------------------------------------------------------------
def find_dir(root: Path, name: str) -> Path:
    """Case-insensitive lookup of a sub-folder."""
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    raise FileNotFoundError(f"Could not find folder '{name}' inside {root}")


def scan(root: Path) -> pd.DataFrame:
    rows = []
    folders = {"test": find_dir(root, "Testing"),
               "trainval": find_dir(root, "Training and Validation")}
    for orig_split, folder in folders.items():
        for cls_name, cls_id in LABELS.items():
            cdir = find_dir(folder, cls_name)
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                rows.append({"path": f.relative_to(root).as_posix(),
                             "label_name": cls_name, "label": cls_id,
                             "orig_split": orig_split})
    return pd.DataFrame(rows)


def phash(im: Image.Image, hash_size: int = 8, highfreq: int = 4) -> str:
    """64-bit perceptual hash (DCT-based) returned as a 16-char hex string."""
    n = hash_size * highfreq
    g = np.asarray(im.convert("L").resize((n, n), Image.LANCZOS), dtype=np.float64)
    low = dctn(g, type=2, norm="ortho")[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    return np.packbits(bits).tobytes().hex()


# ----------------------------------------------------------------------------
# 2-3. verify + hash
# ----------------------------------------------------------------------------
def describe(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    md5s, phs, widths, heights, modes, ok = [], [], [], [], [], []
    for rel in df["path"]:
        fp = root / rel
        try:
            with Image.open(fp) as im:
                im.load()
                widths.append(im.width)
                heights.append(im.height)
                modes.append(im.mode)
                phs.append(phash(im))
            md5s.append(hashlib.md5(fp.read_bytes()).hexdigest())
            ok.append(True)
        except Exception as e:  # corrupt / unreadable file
            print(f"[warn] cannot read {rel}: {e}")
            md5s.append(None); phs.append(None)
            widths.append(None); heights.append(None); modes.append(None)
            ok.append(False)
    df = df.copy()
    df["md5"], df["width"], df["height"], df["mode"] = md5s, widths, heights, modes
    df["readable"] = ok
    df["phash"] = phs
    return df


# ----------------------------------------------------------------------------
# 4. duplicate clusters
# ----------------------------------------------------------------------------
def cluster_duplicates(df: pd.DataFrame, thresh: int):
    n = len(df)
    bits = np.array([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))
                     for h in df["phash"]], dtype=np.uint8)           # (n, 64)
    dist = (bits[:, None, :] != bits[None, :, :]).sum(-1)             # (n, n)

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ii, jj = np.where(np.triu(dist <= thresh, k=1))
    for a, b in zip(ii, jj):
        union(a, b)
    # exact byte duplicates always join, whatever the hash says
    for _, idx in df.groupby("md5").indices.items():
        for k in idx[1:]:
            union(idx[0], k)
    roots = [find(i) for i in range(n)]
    cid = pd.factorize(pd.Series(roots))[0]
    return cid, dist


# ----------------------------------------------------------------------------
# 5. resolve clusters
# ----------------------------------------------------------------------------
def resolve(df: pd.DataFrame):
    keep = pd.Series(True, index=df.index)
    reason = pd.Series("", index=df.index)
    for cid, g in df.groupby("cluster_id"):
        if len(g) == 1:
            continue
        if g["label"].nunique() > 1:                      # label conflict
            keep[g.index] = False
            reason[g.index] = "label_conflict"
            continue
        test_idx = g.index[g["orig_split"] == "test"]
        tv_idx = g.index[g["orig_split"] == "trainval"]
        if len(test_idx) > 0:
            keep_one = test_idx[0]
            for i in g.index:
                if i == keep_one:
                    continue
                keep[i] = False
                reason[i] = ("duplicate_of_test_image" if i in tv_idx
                             else "duplicate_within_test")
        else:
            for i in tv_idx[1:]:                          # keep ONE per cluster
                keep[i] = False
                reason[i] = "duplicate_within_trainval"
    return keep, reason


# ----------------------------------------------------------------------------
# 6. group-aware stratified split
# ----------------------------------------------------------------------------
def split_trainval(tv: pd.DataFrame, val_frac: float, seed: int) -> pd.Series:
    n_splits = max(2, int(round(1 / val_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(tv, tv["label"], groups=tv["cluster_id"]))
    split = pd.Series("train", index=tv.index)
    split.iloc[val_idx] = "val"
    return split


# ----------------------------------------------------------------------------
# 7. report helpers
# ----------------------------------------------------------------------------
def dup_figure(df, dist, out_png, max_pairs=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = []
    for cid, g in df.groupby("cluster_id"):
        if len(g) > 1:
            pairs.append((g.index[0], g.index[1]))
    if not pairs:
        return False
    pairs = pairs[:max_pairs]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(5, 2.4 * len(pairs)))
    axes = np.atleast_2d(axes)
    for r, (a, b) in enumerate(pairs):
        for c, i in enumerate((a, b)):
            with Image.open(ROOT / df.loc[i, "path"]) as im:
                axes[r, c].imshow(im)
            axes[r, c].axis("off")
            axes[r, c].set_title(f'{df.loc[i, "orig_split"]}/{df.loc[i, "label_name"]}\n'
                                 f'd={dist[a, b]}', fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path,
                    help="folder that contains 'Testing' and 'Training and Validation'")
    ap.add_argument("--out", default=Path("data/splits"), type=Path)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--val-frac", default=0.15, type=float)
    ap.add_argument("--hash-thresh", default=4, type=int,
                    help="max pHash Hamming distance (of 64 bits) to call near-duplicates")
    args = ap.parse_args()
    ROOT = args.data_root
    args.out.mkdir(parents=True, exist_ok=True)

    df = scan(ROOT)
    print(f"Found {len(df)} images")
    df = describe(df, ROOT)
    bad = df[~df["readable"]]
    df = df[df["readable"]].reset_index(drop=True)

    df["cluster_id"], dist = cluster_duplicates(df, args.hash_thresh)
    keep, reason = resolve(df)
    df["keep"], df["removed_reason"] = keep, reason

    dup_fig_ok = dup_figure(df, dist, args.out / "duplicate_examples.png")

    kept = df[df["keep"]].copy()
    tv = kept[kept["orig_split"] == "trainval"]
    kept["split"] = ""
    kept.loc[tv.index, "split"] = split_trainval(tv, args.val_frac, args.seed)
    kept.loc[kept["orig_split"] == "test", "split"] = "test"

    cols = ["path", "label", "label_name", "split", "orig_split",
            "cluster_id", "md5", "phash", "width", "height", "mode"]
    for s in ("train", "val", "test"):
        kept[kept["split"] == s][cols].to_csv(args.out / f"{s}.csv", index=False)
    kept[cols].to_csv(args.out / "all_kept.csv", index=False)
    pd.concat([df[~df["keep"]][cols[:2] + ["orig_split", "cluster_id", "removed_reason"]],
               bad[["path", "label_name", "orig_split"]].assign(removed_reason="unreadable")]
              ).to_csv(args.out / "removed.csv", index=False)

    # leakage assertions: no cluster / md5 appears in more than one split
    for key in ("cluster_id", "md5"):
        n_splits_per = kept.groupby(key)["split"].nunique()
        assert (n_splits_per <= 1).all(), f"leakage across splits via {key}"

    summary = {
        "seed": args.seed, "val_frac": args.val_frac, "hash_thresh": args.hash_thresh,
        "n_found": int(len(df) + len(bad)), "n_unreadable": int(len(bad)),
        "n_removed": int((~df["keep"]).sum()),
        "removed_by_reason": df[~df["keep"]]["removed_reason"].value_counts().to_dict(),
        "image_sizes": {f"{w}x{h}": int(c) for (w, h), c in
                        df.groupby(["width", "height"]).size().items()},
        "split_counts": {s: kept[kept["split"] == s]["label_name"].value_counts().to_dict()
                         for s in ("train", "val", "test")},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print("\nCluster-leakage check passed. Files written to", args.out.resolve())
    if not dup_fig_ok:
        print("No duplicate clusters found.")


if __name__ == "__main__":
    main()
