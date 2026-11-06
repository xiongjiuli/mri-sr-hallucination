#!/usr/bin/env python3
"""Verify that the subject-level split is disjoint and report its size."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = json.load(open(os.path.join(ROOT, "data", "processed", "splits.json")))
sets = {k: set(v) for k, v in s.items()}
assert not (sets["train"] & sets["val"]), "train/val overlap"
assert not (sets["train"] & sets["test"]), "train/test overlap"
assert not (sets["val"] & sets["test"]), "val/test overlap"
for k, v in sets.items():
    print(f"{k:5s}: {len(v):3d} subjects")
print("no subject appears in more than one split")
