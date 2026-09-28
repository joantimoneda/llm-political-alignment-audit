##
## Created by: Joan Timoneda
## Date: 17 September 2026
##

"""Grok 3 (March 2026) against Grok 4.3 (July 2026): stance by user identity."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

LEVELS = ["liberal", "mod_liberal", "neutral", "mod_conservative", "conservative"]
LABELS = ["lib", "m-lib", "neu", "m-con", "con"]
GENS = [("grok-3 (2026-03, rescored)", "Grok 3 (February 2026)"), ("grok-4.3 (2026-07)", "Grok 4.3 (July 2026)")]
ROWS = [("abortion", "stance (0=illegal … 10=legal)", True)]
C_GENUINE, C_AT5, C_TEXT, C_MUTED, C_BAND = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#f0f0ec"


def draw(numbers, out_path):
    d = pd.read_csv(numbers)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), sharey=True)
    axes = np.atleast_2d(axes)
    xs = np.arange(len(LEVELS))
    for i, (topic, ylabel, band) in enumerate(ROWS):
        for j, (gen_key, gen_name) in enumerate(GENS):
            ax = axes[i, j]
            s = d[(d.generation == gen_key) & (d.scoring_topic == topic)].set_index("cue_level").reindex(LEVELS)
            gen = s.stance_genuine.to_numpy(dtype=float)
            pct = s.pct_answering.to_numpy(dtype=float)
            at5 = np.where(np.isnan(gen), 5.0, pct * np.nan_to_num(gen) + (1 - pct) * 5.0)
            if band:
                ax.axhspan(4, 6, color=C_BAND, zorder=0)
            ax.axhline(5, color="#b8b8b4", lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.plot(xs, at5, color=C_AT5, lw=2.2, ls="--", marker="o", ms=7, zorder=2)
            ok = ~np.isnan(gen)
            ax.plot(xs[ok], gen[ok], color=C_GENUINE, lw=2.2, marker="o", ms=7, zorder=3)
            ax.set_xlim(-0.5, len(LEVELS) - 0.5)
            ax.set_ylim(-0.5, 10.5)
            ax.set_yticks(range(0, 11, 2))
            ax.set_xticks(xs)
            ax.set_xticklabels(LABELS, fontsize=9.5, color=C_TEXT)
            for x, p in zip(xs, pct):
                ax.text(x, -1.85, f"{100 * p:.0f}", ha="center", va="top", fontsize=9.5, color=C_GENUINE, clip_on=False)
            ax.grid(axis="y", color="#ececea", lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.spines["left"].set_color("#b8b8b4"); ax.spines["bottom"].set_color("#b8b8b4")
            ax.tick_params(length=0)
            ax.set_title(gen_name,
                         loc="left", fontsize=12, fontweight="bold", color=C_TEXT, pad=8)
        axes[i, 0].set_ylabel(ylabel, fontsize=10, color=C_TEXT)
    fig.suptitle("Grok across releases — abortion stance by stated user ideology · same script, same judge",
                 x=0.045, ha="left", fontsize=13, fontweight="bold", color=C_TEXT, y=0.985)
    handles = [Line2D([0], [0], color=C_AT5, lw=2.2, ls="--", marker="o", ms=7),
               Line2D([0], [0], color=C_GENUINE, lw=2.2, marker="o", ms=7),
               Line2D([0], [0], color="#b8b8b4", lw=1.0, ls=(0, (4, 3)))]
    labels = ["TRANSMITTED — refusals coded 5 (100% of answers)",
              "WHEN IT ANSWERS — genuine positions only; blue % below = share of answers behind this line",
              "scale midpoint (5); shaded band = preregistered neutral range [4, 6]"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.045, 0.94), frameon=False,
               fontsize=9.5, handlelength=2.6, labelspacing=0.5)
    fig.text(0.045, 0.012,
             "Blue numbers: % of the cell's answers that took a position; the remainder are refusals. 150 answers per identity in each arm, "
             "three turns pooled; single judge (GPT-5.5); the February archive has no no-persona conversations.",
             fontsize=9, color=C_MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.83))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--out", default="figures/grok_releases.png")
    a = ap.parse_args()
    print("wrote", draw(a.numbers, a.out))
