"""
fire_common.py - shared code for train.py and evaluate.py
Smoke and Fire Recognition in Forest Images.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             accuracy_score)
from torch.utils.data import Dataset
from torchvision import models, transforms

IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLASS_NAMES = ["nofire", "fire"]          # label 1 = fire (positive class)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def make_transforms(train: bool, aug: str = "basic"):
    """aug: 'none' | 'basic' | 'strong' (only used when train=True)."""
    norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if not train or aug == "none":
        return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))] + norm)
    if aug == "basic":
        t = [transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
             transforms.RandomHorizontalFlip()]
    elif aug == "strong":
        t = [transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(15),
             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)]
    else:
        raise ValueError(aug)
    return transforms.Compose(t + norm)


class FireDataset(Dataset):
    def __init__(self, csv_path, data_root, transform, limit=None):
        self.df = pd.read_csv(csv_path)
        if limit:                                   # smoke tests only
            self.df = self.df.groupby("label", group_keys=False).head(limit // 2)
            self.df = self.df.reset_index(drop=True)
        self.root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.root / row["path"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, torch.tensor(float(row["label"]))


# ----------------------------------------------------------------------------
# models  (all output ONE logit: positive class = fire)
# ----------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """Baseline: 4 conv blocks trained from scratch (~0.4M parameters)."""

    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64),
                                      block(64, 128), block(128, 256))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


class _Squeeze(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).squeeze(1)


def build_model(arch: str, pretrained: bool):
    """Returns (model, head_parameters)."""
    w = "DEFAULT" if pretrained else None
    if arch == "cnn":
        m = SmallCNN()
        return m, list(m.parameters())
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
        head = list(m.classifier.parameters())
        return _Squeeze(m), head
    if arch == "resnet18":
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return _Squeeze(m), list(m.fc.parameters())
    raise ValueError(f"unknown arch {arch}")


def head_param_ids(model, arch):
    inner = model.m if isinstance(model, _Squeeze) else model
    if arch == "mobilenet_v3_small":
        return {id(p) for p in inner.classifier.parameters()}
    if arch == "resnet18":
        return {id(p) for p in inner.fc.parameters()}
    return {id(p) for p in inner.parameters()}


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device))).cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys).astype(int)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0)
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tp = int(((pred == 1) & (y == 1)).sum())
    return {"threshold": float(thr), "accuracy": float(accuracy_score(y, pred)),
            "fire_precision": float(prec), "fire_recall": float(rec), "fire_f1": float(f1),
            "ap": float(average_precision_score(y, p)),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def choose_threshold(y, p, target_recall=0.95):
    """Pick the decision threshold ON VALIDATION DATA ONLY.
    Returns the highest threshold whose fire recall >= target_recall
    (fewest false alarms subject to the recall target)."""
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys == 1)
    recall = tp / max(1, (y == 1).sum())
    ok = np.where(recall >= target_recall)[0]
    if len(ok) == 0:
        return 0.5
    k = ok[0]
    # threshold just at the probability of the k-th ranked sample
    return float(ps[k])
