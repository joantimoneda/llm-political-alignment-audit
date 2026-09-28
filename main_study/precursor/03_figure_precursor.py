##
## Created by: Joan Timoneda
## Date: 12 September 2026
##

"""Six-panel figure of the pre-question lean, one per topic."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCORED = HERE / "data" / "lean_scored.csv"
RESULTS = HERE / "results"

MODELS = ["gpt", "claude", "grok", "deepseek", "mistral", "gemma"]
CONFIG = {
    "abortion": dict(levels=["liberal", "mod_liberal", "neutral", "mod_conservative", "conservative"],
                     labels=["lib", "m-lib", "neu", "m-con", "con"],
                     ylabel="lean of the reply (0=illegal … 10=legal)",
                     title="Abortion — lean of the replies before any question is asked · 6 developers · n=50 per point"),
    "catalan": dict(levels=["union", "mod_union", "neutral", "mod_independence", "independence"],
                    labels=["union", "m-union", "neutral", "m-indep", "indep"],
                    ylabel="lean of the reply (0=unionist … 10=independence)",
                    title="Catalan independence — lean of the replies before any question is asked · 6 developers · n=50 per point"),
    "climate": dict(levels=["denier", "mod_skeptic", "neutral", "mod_accept", "accept"],
                    labels=["denier", "m-skeptic", "neutral", "m-accept", "accept"],
                    ylabel="lean of the reply (0=hoax … 10=IPCC-consistent)",
                    title="Climate change — lean of the replies before any question is asked · 6 developers · n=50 per point"),
    "nazism": dict(levels=["sympathetic", "neutral"],
                   labels=["sympathetic", "neutral"],
                   ylabel="lean of the reply (0=exculpates … 10=condemns)",
                   title="Nazism — lean of the replies before any question is asked · 6 developers · n=50 per point"),
    "control": dict(levels=["anti", "neutral", "pro"],
                    labels=["anti", "neutral", "pro"],
                    ylabel="lean of the reply (0=anti … 10=pro pineapple pizza)",
                    title="Control (pineapple on pizza) — lean of the replies before any question is asked · 6 developers · n=50 per point"),
}
STEP_COLORS = ["#a9c9f0", "#6ea3e6", "#2a78d6", "#0f3f8a"]
STEP_LABELS = ["reply to identity statement 1", "reply to identity statement 2",
               "reply to identity statement 3", "reply to identity statement 4 (the user's summary)"]
C_TEXT, C_MUTED = "#0b0b0b", "#52514e"


def cell_table(df, topic):
    d = df[(df.topic == topic) & df.lean_mean.notna()]
    return d.groupby(["model", "cue_level", "cue_step"]).agg(lean=("lean_mean", "mean"),
                                                             n=("lean_mean", "size")).reset_index()


def draw(tab, topic, out_path, judges_note):
    cfg = CONFIG[topic]
    levels, labels = cfg["levels"], cfg["labels"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4), sharey=True)
    xs = np.arange(len(levels))
    for ax, m in zip(axes.flat, MODELS):
        t = tab[tab.model == m]
        ax.axhline(5, color="#b8b8b4", lw=1.0, ls=(0, (4, 3)), zorder=1)
        for s, (col, lab) in enumerate(zip(STEP_COLORS, STEP_LABELS), start=1):
            ys = np.array([t[(t.cue_level == l) & (t.cue_step == s)].lean.mean() for l in levels])
            ok = ~np.isnan(ys)
            if ok.any():
                ax.plot(xs[ok], ys[ok], color=col, lw=2.0, marker="o", ms=6, zorder=2 + s)
        if t.empty:
            ax.text(0.5, 0.5, "not scored", transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=C_MUTED, style="italic")
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_yticks(range(0, 11, 2))
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9.5, color=C_TEXT)
        ax.grid(axis="y", color="#ececea", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#b8b8b4"); ax.spines["bottom"].set_color("#b8b8b4")
        ax.tick_params(length=0)
        ax.set_title(m, loc="left", fontsize=13, fontweight="bold", color=C_TEXT, pad=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(cfg["ylabel"], fontsize=10, color=C_TEXT)
    fig.suptitle(cfg["title"], x=0.045, ha="left", fontsize=14, fontweight="bold", color=C_TEXT, y=0.985)
    handles = [Line2D([0], [0], color=c, lw=2.0, marker="o", ms=6) for c in STEP_COLORS] + \
              [Line2D([0], [0], color="#b8b8b4", lw=1.0, ls=(0, (4, 3)))]
    fig.legend(handles, STEP_LABELS + ["5 = no discernible lean of the reply's own"],
               loc="upper left", bbox_to_anchor=(0.045, 0.955), frameon=False, fontsize=10,
               handlelength=2.6, labelspacing=0.4, ncol=2)
    fig.text(0.045, 0.012,
             "Mean lean of the system's reply to each of the user's four identity statements, before the direct question.\n"
             f"{judges_note} Exploratory, not preregistered.",
             fontsize=9.5, color=C_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.055, 1, 0.86))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    return out_path


def main(a):
    df = pd.read_csv(a.scored, low_memory=False)
    judges = [c[5:] for c in df.columns if c.startswith("lean_") and c not in ("lean_mean", "lean_spread")]
    names = {"gpt": "GPT-5.5", "claude": "Claude Opus 4.8"}
    judges_note = ("Rated by " + " and ".join(names.get(j, j) for j in judges)
                   + (", mean of the two." if len(judges) > 1 else "."))
    out = a.out or (RESULTS / f"precursor_{a.topic}_six_models.png")
    print("wrote", draw(cell_table(df, a.topic), a.topic, out, judges_note))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Six-panel figure of the pre-question lean.")
    ap.add_argument("--scored", default=str(SCORED))
    ap.add_argument("--topic", default="abortion", choices=list(CONFIG))
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
