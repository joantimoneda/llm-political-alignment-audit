##
## Created by: Joan Timoneda
## Date: 17 September 2026
##

"""Grok 3 against Grok 4.3: stance by conversation turn."""

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
TURNS = [(0, "test question", "#a9c8ee"), (1, "first press", "#5b96dd"), (2, "second press", "#1f4f9a")]
C_TEXT, C_MUTED, C_BAND = "#0b0b0b", "#52514e", "#f0f0ec"
MIN_N = 5


def draw(numbers, out_path):
    d = pd.read_csv(numbers)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.6), sharey=True)
    xs = np.arange(len(LEVELS))
    for j, (gen_key, gen_name) in enumerate(GENS):
        ax_stance = axes[j]
        for step, tname, color in TURNS:
            s = d[(d.generation == gen_key) & (d.pressure_step == step)].set_index("cue_level").reindex(LEVELS)
            stance = s.genuine.to_numpy(dtype=float)
            n_st = s.n_stance.to_numpy(dtype=float)
            ok = (~np.isnan(stance)) & (n_st >= MIN_N)
            ax_stance.plot(xs[ok], stance[ok], color=color, lw=2.2, marker="o", ms=6.5, zorder=3)
            few = (~np.isnan(stance)) & (n_st < MIN_N) & (n_st > 0)
            ax_stance.plot(xs[few], stance[few], color=color, marker="o", ms=6.5, ls="none", mfc="white", mew=1.8, zorder=3)
            share = 100 * s.pct_answering.to_numpy(dtype=float)
            for x, sh in zip(xs, share):
                ax_stance.text(x, -1.6 - 0.75 * step, f"{sh:.0f}", ha="center", va="top", fontsize=9, color=color, clip_on=False)
        ax_stance.set_title(gen_name, loc="left", fontsize=13, fontweight="bold", color=C_TEXT, pad=8)
        ax_stance.axhspan(4, 6, color=C_BAND, zorder=0)
        ax_stance.axhline(5, color="#b8b8b4", lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax_stance.set_ylim(-0.5, 10.5); ax_stance.set_yticks(range(0, 11, 2))
        for ax in (ax_stance,):
            ax.set_xlim(-0.5, len(LEVELS) - 0.5)
            ax.set_xticks(xs); ax.set_xticklabels(LABELS, fontsize=9.5, color=C_TEXT)
            ax.grid(axis="y", color="#ececea", lw=0.8, zorder=0); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.spines["left"].set_color("#b8b8b4"); ax.spines["bottom"].set_color("#b8b8b4")
            ax.tick_params(length=0)
    axes[0].set_ylabel("stance (0=illegal … 10=legal)", fontsize=10, color=C_TEXT)
    fig.suptitle("Grok across releases — abortion stance by turn · same script, same judge",
                 x=0.045, ha="left", fontsize=13.5, fontweight="bold", color=C_TEXT, y=0.985)
    handles = [Line2D([0], [0], color=c, lw=2.2, marker="o", ms=6.5) for _, _, c in TURNS] + \
              [Line2D([0], [0], color="#1f4f9a", marker="o", ms=6.5, ls="none", mfc="white", mew=1.8)]
    labels = [t for _, t, _ in TURNS] + ["fewer than five answers (mean shown, not joined)"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.045, 0.93), frameon=False,
               fontsize=9.5, ncol=4, handlelength=2.4, columnspacing=1.6)
    fig.text(0.045, 0.012,
             "Lines: mean stance among answers that took a position; refusals excluded; a line joins identities with at least five such answers.\n"
             "Numbers below each identity: % of that turn's 50 answers that took a position (test question, first press, second press, top to bottom).\n"
             "Single judge (GPT-5.5) for both releases; the February archive has no no-persona conversations.",
             fontsize=8.8, color=C_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 0.86))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--out", default="figures/grok_releases_by_turn.png")
    a = ap.parse_args()
    print("wrote", draw(a.numbers, a.out))
