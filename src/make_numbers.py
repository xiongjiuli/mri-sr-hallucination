#!/usr/bin/env python3
"""Generate paper/numbers.tex from result files.

Every number printed in the paper is produced here so that the manuscript
cannot drift from the experiments.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "numbers.tex")

MODELS = {
    "bic": "bicubic",
    "srres": "srresnet_l1",
    "rrdb": "rrdb_l1",
    "rrdbmkda": "rrdb_l1_mkda",
    "rrdbdsc": "rrdb_dsc",
    "perc": "rrdb_perc",
    "trust": "rrdb_dsc_mkda",
    "trustg": "rrdb_dsc_mkda_gate",
    "shr": "rrdb_l1_s90",
}


def load_summary(tag):
    p = os.path.join(RES, f"summary_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def load_cop(tag):
    p = os.path.join(RES, f"cop_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(v, nd=2):
    return "--" if v is None else f"{v:.{nd}f}"


def main():
    macros = {}
    meta = json.load(open(os.path.join(ROOT, "data", "processed", "meta.json")))
    splits = json.load(open(os.path.join(ROOT, "data", "processed", "splits.json")))
    macros["numTrainSubjects"] = str(len(splits["train"]))
    macros["numValSubjects"] = str(len(splits["val"]))
    macros["numTestSubjects"] = str(len(splits["test"]))
    macros["numTrainSlices"] = f"{len(meta['train']):,}".replace(",", "{,}")
    macros["numValSlices"] = f"{len(meta['val']):,}".replace(",", "{,}")
    macros["numTestSlices"] = f"{len(meta['test']):,}".replace(",", "{,}")
    macros["numParams"] = "3.21\\,M"
    macros["numTorchVersion"] = "2.12"
    macros["numSteps"] = "5{,}000"

    summaries, cops, csvs = {}, {}, {}
    for key, tag in MODELS.items():
        summaries[key] = load_summary(tag)
        cops[key] = load_cop(tag)
        p = os.path.join(RES, f"metrics_{tag}.csv")
        if os.path.exists(p):
            with open(p) as f:
                csvs[key] = list(csv.DictReader(f))

    for key in MODELS:
        s, c = summaries[key], cops[key]
        for metric, macro in [("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS"),
                              ("hfen", "HFEN"), ("edge_ssim", "EDGE"), ("leak", "LEAK")]:
            val = None
            if s and "id" in s and metric in s["id"]:
                val = s["id"][metric]["mean"] if isinstance(s["id"][metric], dict) else s["id"][metric]
            macros[f"{macro}{key}"] = fmt(val, 2 if metric == "psnr" else 3 if metric == "lpips" else 4)
        for metric, macro in [("he", "HE"), ("vr", "VR"), ("mae", "MAE")]:
            val = c["summary"][metric] if c else None
            macros[f"{macro}{key}"] = fmt(val, 4)
        for deg, pre in [("blur1.4", "OODblur"), ("bicubic", "OODbic"), ("noise", "OODnoise")]:
            val = None
            if s and deg in s and "psnr" in s[deg]:
                val = s[deg]["psnr"]["mean"] if isinstance(s[deg]["psnr"], dict) else s[deg]["psnr"]
            macros[f"{pre}{key.capitalize()}"] = fmt(val, 2)

    def val(key, deg="id", metric="psnr"):
        s = summaries.get(key)
        if not s or deg not in s or metric not in s[deg]:
            return None
        v = s[deg][metric]
        return v["mean"] if isinstance(v, dict) else v

    def copval(key, metric="he"):
        c = cops.get(key)
        return c["summary"][metric] if c else None

    base_psnr = val("rrdb"); base_he = copval("rrdb")
    for key in ["rrdbmkda", "rrdbdsc", "trust", "trustg"]:
        p, h = val(key), copval(key)
        macros[f"DPSNR{key}"] = fmt(None if (p is None or base_psnr is None) else p - base_psnr, 3)
        macros[f"DHE{key}"] = fmt(None if (h is None or base_he is None) else h - base_he, 4)
    if base_he and copval("trust") is not None:
        macros["HEreductionPct"] = fmt(100 * (base_he - copval("trust")) / base_he, 1)
    else:
        macros["HEreductionPct"] = "--"
    if base_he and copval("trustg") is not None:
        macros["HEgatePct"] = fmt(100 * (base_he - copval("trustg")) / base_he, 1)
    else:
        macros["HEgatePct"] = "--"
    if base_he and copval("perc") is not None:
        macros["HEpercPct"] = fmt(100 * (copval("perc") - base_he) / base_he, 1)
    else:
        macros["HEpercPct"] = "--"
    if base_he and copval("rrdbdsc") is not None:
        macros["HEdscPct"] = fmt(100 * (base_he - copval("rrdbdsc")) / base_he, 1)
    else:
        macros["HEdscPct"] = "--"
    if val("trustg") is not None and val("trust") is not None:
        macros["PSNRdropGate"] = fmt(val("trust") - val("trustg"), 3)
    else:
        macros["PSNRdropGate"] = "--"
    if val("rrdb") is not None and val("trust") is not None:
        macros["PSNRdrop"] = fmt(val("rrdb") - val("trust"), 3)
    else:
        macros["PSNRdrop"] = "--"
    # OOD gain: MKDA vs RRDB under bicubic degradation
    a, b = val("rrdb", "blur1.4"), val("rrdbmkda", "blur1.4")
    macros["OODgain"] = fmt(None if (a is None or b is None) else b - a, 2)

    # shrinkage and gate controls
    if copval("shr") is not None and base_he:
        macros["HEshrinkPct"] = fmt(100 * (base_he - copval("shr")) / base_he, 1)
    else:
        macros["HEshrinkPct"] = "--"
    if val("shr") is not None and val("rrdb") is not None:
        macros["PSNRshrink"] = fmt(val("rrdb") - val("shr"), 3)
    else:
        macros["PSNRshrink"] = "--"
    if val("rrdb") is not None and val("trust", "blur1.4") is not None:
        macros["OODgainTrust"] = fmt(val("trust", "blur1.4") - val("rrdb", "blur1.4"), 2)
    else:
        macros["OODgainTrust"] = "--"
    sweep = load_json(os.path.join(RES, "gate_sweep.json"))
    if sweep:
        g1 = sweep["betas"].get("1.0")
        uni = sweep.get("uniform", {})
        if g1:
            macros["HEgateOne"] = fmt(g1["he"], 4)
            macros["PSNRgateOne"] = fmt(g1["psnr"], 3)
            macros["SSIMgateOne"] = fmt(g1["ssim"], 4)
            best = min(uni.items(), key=lambda kv: abs(kv[1]["he"] - g1["he"])) if uni else None
            if best:
                macros["HEuniformOne"] = fmt(best[1]["he"], 4)
                macros["PSNRuniformOne"] = fmt(best[1]["psnr"], 3)
                macros["SSIMuniformOne"] = fmt(best[1]["ssim"], 4)
    for k in ["HEgateOne", "PSNRgateOne", "SSIMgateOne", "HEuniformOne", "PSNRuniformOne", "SSIMuniformOne"]:
        macros.setdefault(k, "--")

    # correlations across model means
    keys = [k for k in MODELS if summaries[k] and cops[k]]
    if len(keys) >= 3:
        from scipy.stats import spearmanr
        psn = [val(k) for k in keys]; lp = [val(k, metric="lpips") for k in keys]
        he = [copval(k) for k in keys]
        ok = [i for i in range(len(keys)) if None not in (psn[i], he[i])]
        macros["corrPSNRHE"] = fmt(float(spearmanr([psn[i] for i in ok], [he[i] for i in ok]).statistic), 2)
        ok2 = [i for i in range(len(keys)) if None not in (lp[i], he[i])]
        macros["corrLPIPSHE"] = fmt(float(spearmanr([lp[i] for i in ok2], [he[i] for i in ok2]).statistic), 2)
    else:
        macros["corrPSNRHE"] = macros["corrLPIPSHE"] = "--"

    # paired Wilcoxon on per-slice PSNR and COP HE
    def per_slice_he(tag):
        c = load_cop(tag)
        if not c:
            return {}
        d = defaultdict(list)
        for r in c["patches"]:
            d[r["index"]].append(r["he"])
        return {k: float(np.mean(v)) for k, v in d.items()}

    def p_value(a, b):
        if not a or not b:
            return "--"
        common = sorted(set(a) & set(b))
        if len(common) < 8:
            return "--"
        from scipy.stats import wilcoxon
        return f"{wilcoxon([a[i] for i in common], [b[i] for i in common]).pvalue:.1e}"

    def per_slice_psnr(tag):
        rows = csvs.get(tag)
        if not rows:
            return {}
        return {int(r["index"]): float(r["psnr"]) for r in rows if r["deg"] == "id"}

    macros["pTrustRRDBpsnr"] = p_value(per_slice_psnr("rrdb_l1"), per_slice_psnr("rrdb_dsc_mkda"))
    macros["pTrustRRDBhe"] = p_value(per_slice_he("rrdb_l1"), per_slice_he("rrdb_dsc_mkda"))
    macros["pPercRRDBhe"] = p_value(per_slice_he("rrdb_l1"), per_slice_he("rrdb_perc"))
    macros["pShrinkRRDBhe"] = p_value(per_slice_he("rrdb_l1"), per_slice_he("rrdb_l1_s90"))

    corr = {"psnr_he": float(macros["corrPSNRHE"]) if macros["corrPSNRHE"] != "--" else None,
            "lpips_he": float(macros["corrLPIPSHE"]) if macros["corrLPIPSHE"] != "--" else None}
    with open(os.path.join(RES, "corr.json"), "w") as f:
        json.dump(corr, f, indent=1)

    with open(OUT, "w") as f:
        f.write("% Auto-generated by src/make_numbers.py -- do not edit.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {OUT} ({len(macros)} macros)")


if __name__ == "__main__":
    main()
