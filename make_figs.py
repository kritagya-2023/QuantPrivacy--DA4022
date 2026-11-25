"""Builds the mid-project figure straight from results/results.json.

    python make_figs.py --results results/results.json --out results/fig_midpoint.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY = "#21295C"
BLUE = "#1C7293"
TEAL = "#02C39A"
RUST = "#C0492B"
GREY = "#6B7280"


def build(results, out_path):
    arms = results["arms"]
    schemes = [s for s in ["fp32", "fp16", "int8", "int4"] if s in arms]
    labels = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8", "int4": "INT4"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))

    # -- panel 1: attack strength across precisions -------------------------
    attack_names = ["loss", "confidence", "entropy"]
    colors = [NAVY, BLUE, TEAL]
    x = np.arange(len(schemes))
    w = 0.26
    for i, (a, c) in enumerate(zip(attack_names, colors)):
        vals = [arms[s]["attacks"][a]["auc"] for s in schemes]
        ax1.bar(x + (i - 1) * w, vals, w, label=a.capitalize(), color=c)
    ax1.axhline(0.5, color=GREY, ls="--", lw=1)
    ax1.text(len(schemes) - 0.45, 0.505, "chance", color=GREY, fontsize=8)
    ax1.set_xticks(x, [labels[s] for s in schemes])
    ax1.set_ylabel("Attack AUC")
    ax1.set_title("Membership inference strength vs. weight precision", fontsize=10)
    ax1.set_ylim(0.45, max(0.75, max(arms[s]["attacks"]["loss"]["auc"] for s in schemes) + 0.05))
    ax1.legend(frameon=False, fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # -- panel 2: privacy / utility / footprint trade-off -------------------
    foot = [arms[s]["footprint_mb"] for s in schemes]
    tpr = [arms[s]["attacks"]["loss"]["tpr@1%fpr"] * 100 for s in schemes]
    acc = [arms[s]["test_acc"] * 100 for s in schemes]

    ax2.plot(foot, tpr, "-o", color=RUST, lw=2, label="TPR @ 1% FPR (worst-case leakage)")
    for f, t, s in zip(foot, tpr, schemes):
        ax2.annotate(labels[s], (f, t), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax2.margins(x=0.14, y=0.14)
    ax2.set_xlabel("Weight footprint (MB)")
    ax2.set_ylabel("TPR @ 1% FPR (%)", color=RUST)
    ax2.tick_params(axis="y", labelcolor=RUST)
    ax2.spines[["top"]].set_visible(False)

    ax3 = ax2.twinx()
    ax3.plot(foot, acc, "--s", color=BLUE, lw=1.6, label="Test accuracy")
    ax3.set_ylabel("Test accuracy (%)", color=BLUE)
    ax3.tick_params(axis="y", labelcolor=BLUE)
    ax3.spines[["top"]].set_visible(False)
    ax2.set_title("Privacy–utility–memory trade-off", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--out", default="results/fig_midpoint.png")
    a = ap.parse_args()
    with open(a.results) as f:
        build(json.load(f), a.out)
