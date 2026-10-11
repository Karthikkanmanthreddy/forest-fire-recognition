"""
colour_baseline.py - simple classical baseline / dataset diagnostic.

Logistic regression on global colour statistics (mean grey, mean R,G,B).
It shows how much of the task is solvable from global colour alone.

By default it is trained on train.csv and evaluated on val.csv ONLY.
Pass --also-test to evaluate once on the locked test set. That evaluation was
added AFTER the main experiments as a diagnostic and must be reported as such;
it is not used for any model selection.

    python colour_baseline.py
    python colour_baseline.py --also-test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.metrics import precision_recall_fscore_support

FEATURE_SETS = {"brightness only": [0], "mean R,G,B": [1, 2, 3]}


def features(csv, root):
    df = pd.read_csv(csv)
    X = []
    for p in df.path:
        with Image.open(Path(root) / p) as im:
            a = np.asarray(im.convert("RGB"), dtype=float)
        X.append([a.mean(), *a.mean(axis=(0, 1))])
    return np.array(X), df.label.values


def evaluate(y, p):
    pred = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                    pos_label=1, zero_division=0)
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, pred)), "fire_precision": float(pr),
            "fire_recall": float(rc), "fire_f1": float(f1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw/Dataset")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="results")
    ap.add_argument("--also-test", action="store_true")
    args = ap.parse_args()

    sp = Path(args.splits)
    Xtr, ytr = features(sp / "train.csv", args.data_root)
    Xva, yva = features(sp / "val.csv", args.data_root)
    Xte = yte = None
    if args.also_test:
        Xte, yte = features(sp / "test.csv", args.data_root)

    res = {"note": "diagnostic baseline; test evaluation (if present) added post hoc"}
    for name, cols in FEATURE_SETS.items():
        clf = LogisticRegression(max_iter=1000).fit(Xtr[:, cols], ytr)
        res[name] = {"val": evaluate(yva, clf.predict_proba(Xva[:, cols])[:, 1])}
        if args.also_test:
            res[name]["test"] = evaluate(yte, clf.predict_proba(Xte[:, cols])[:, 1])
        print(name, json.dumps(res[name], indent=1))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "colour_baseline.json").write_text(json.dumps(res, indent=2))
    print("saved", (out / "colour_baseline.json").resolve())


if __name__ == "__main__":
    main()
