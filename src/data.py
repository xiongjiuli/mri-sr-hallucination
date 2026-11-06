"""Datasets for training and full-slice evaluation."""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import degrade

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")


class TrainDataset(Dataset):
    """Random HR patches from subject-disjoint training slices."""

    def __init__(self, split: str = "train", patch: int = 192, scale: int = 4,
                 sigmas: tuple[float, ...] = (0.8,), root: str = PROC, augment: bool = True):
        self.arr = np.load(os.path.join(root, f"{split}.npy"), mmap_mode="r")
        self.patch = patch
        self.scale = scale
        self.sigmas = tuple(sigmas)
        self.augment = augment
        self.meta = json.load(open(os.path.join(root, "meta.json")))[split]

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        hr = np.array(self.arr[i], dtype=np.float32)
        h, w = hr.shape
        p = self.patch
        if self.augment:
            y = np.random.randint(0, h - p + 1)
            x = np.random.randint(0, w - p + 1)
        else:
            y = (h - p) // 2
            x = (w - p) // 2
        crop = hr[y:y + p, x:x + p]
        t = torch.from_numpy(np.ascontiguousarray(crop))[None]
        if self.augment:
            if np.random.rand() < 0.5:
                t = torch.flip(t, dims=[2])
            if np.random.rand() < 0.5:
                t = torch.flip(t, dims=[1])
            k = np.random.randint(0, 4)
            if k:
                t = torch.rot90(t, k, dims=[1, 2])
        return {"hr": t, "index": i}


class EvalDataset(Dataset):
    """Full axial slices, including the subject id for per-subject analysis."""

    def __init__(self, split: str = "test", root: str = PROC, max_items: int | None = None,
                 stride: int = 1):
        self.arr = np.load(os.path.join(root, f"{split}.npy"), mmap_mode="r")
        self.meta = json.load(open(os.path.join(root, "meta.json")))[split]
        self.idx = list(range(0, len(self.arr), stride))
        if max_items:
            self.idx = self.idx[:max_items]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        hr = torch.from_numpy(np.array(self.arr[j], dtype=np.float32))[None]
        return {"hr": hr, "subject": self.meta[j]["subject"], "slice": self.meta[j]["slice"], "index": j}
