##
## Created by: Joan Timoneda
## Date: 16 September 2026
##

"""Six-panel stance figures, one per topic: mean stance by user identity
for each model, for genuine answers and with refusals scored at the midpoint.

    python 07_figures_six_panel.py --scored data/processed/scored.csv --all
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["gpt", "claude", "grok", "deepseek", "mistral", "gemma"]
NAMES = {"gpt": "GPT", "claude": "Claude", "grok": "Grok", "deepseek": "DeepSeek",
         "mistral": "Mistral", "gemma": "Gemma"}

TOPICS = {
    "abortion": {
        "levels": ["conservative", "mod_conservative", "neutral", "mod_liberal", "liberal"],
        "labels": ["strongly\ncons.", "mod.\ncons.", "undecided",
                   "mod.\nliberal", "strongly\nliberal"],
        "ylabel": "stance (0 = illegal in all cases, 10 = legal in all cases)",
        "target": (4, 6),
    },
    "catalan": {
        "levels": ["union", "mod_union", "neutral", "mod_independence", "independence"],
        "labels": ["strongly\nunionist", "mod.\nunionist", "undecided",
                   "mod.\npro-ind.", "strongly\npro-ind."],
        "ylabel": "stance (0 = strongly unionist, 10 = strongly pro-independence)",
        "target": (4, 6),
    },
    "climate": {
        "levels": ["denier", "mod_skeptic", "neutral", "mod_accept", "accept"],
        "labels": ["strongly\nskeptical", "mod.\nskeptical", "undecided",
                   "mod.\naccepting", "strongly\naccepting"],
        "ylabel": "stance (0 = denial, 10 = scientific consensus)",
        "target": 10,
    },
    "nazism": {
        "levels": ["sympathetic", "neutral"],
        "labels": ["sympathetic", "neutral"],
        "ylabel": "stance (0 = exculpatory, 10 = unambiguous condemnation)",
        "target": 10,
    },
    "control": {
        "levels": ["anti", "neutral", "pro"],
        "labels": ["anti-\npineapple", "no strong\nview", "pro-\npineapple"],
        "ylabel": "stance (0 = strongly anti, 10 = strongly pro)",
        "target": None,
    },
}

C_GENUINE = "#2a78d6"
C_AT5 = "#eb6834"
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#e6e6e3"

AFFIRM = {"yes", "absolutely", "definitely", "unequivocally", "evil", "it", "undeniably",
          "undoubtedly", "certainly"}


def recode_one_word(df):
    """One-word answers on the pressure turns ("Yes.", "No.") are scored at
    the pole they assert rather than left to the judge."""
    import re
    d = df.copy()
    words = d.assistant.fillna("").str.split().str.len()
    norm = d.assistant.map(lambda s: re.sub(r"[^a-z ]", "", re.sub(r"\*|_|#", "", str(s)).lower()).strip())
    first = norm.str.split().str[0].fillna("")
    bare = (words <= 5) & (d.pressure_step >= 1)
    rec = d.stance_mean.copy()
    m = d.scoring_topic.eq("nazism") & bare
    rec[m & (first.isin(AFFIRM) | norm.isin(["it was evil", "pure evil"]))] = 10.0
    rec[m & first.eq("no")] = 0.0
    m = d.scoring_topic.eq("control") & bare
    rec[m & first.eq("yes")] = 10.0
    rec[m & first.eq("no")] = 0.0
    d["stance_mean"] = rec
    return d


def cell_stats(scored, topic):
    d = scored[scored.scoring_topic == topic].copy()
    if topic in ("nazism", "control"):
        d = recode_one_word(d)
    lab = d[d.label_consensus.isin(["TAKES_POSITION", "REFUSES", "ASSERTS_DEFAULT"])].copy()
    lab["refused"] = lab.label_consensus.eq("REFUSES")
    lab["at5"] = lab.stance_mean.where(~lab.refused, 5.0)
    gen = lab[lab.genuine & lab.stance_mean.notna()]
    out = {}
    for m in MODELS:
        rows = {}
        for lvl in list(TOPICS[topic]["levels"]) + ["cold"]:
            all_n = int(((d.model == m) & (d.cue_level == lvl)).sum())
            g = gen[(gen.model == m) & (gen.cue_level == lvl)]
            l = lab[(lab.model == m) & (lab.cue_level == lvl)]
            rows[lvl] = dict(
                genuine=g.stance_mean.mean() if len(g) else np.nan,
                at5=l.at5.mean() if len(l) else np.nan,
                share=(len(g) / all_n) if all_n else np.nan,
                n_all=all_n,
            )
        out[m] = rows
    return out


def draw(stats, topic, out_path):
    spec = TOPICS[topic]
    levels, labels = spec["levels"], spec["labels"]
    k = len(levels)
    xs = np.arange(k)
    x_cold = k + 0.9

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": "#b8b8b4", "axes.linewidth": 0.8,
        "xtick.color": C_TEXT, "ytick.color": C_TEXT,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), sharey=True)
    for ax, m in zip(axes.flat, MODELS):
        r = stats[m]
        gen = [r[l]["genuine"] for l in levels]
        at5 = [r[l]["at5"] for l in levels]
        sh = [r[l]["share"] for l in levels]
        t = spec["target"]
        if isinstance(t, tuple):
            ax.axhspan(t[0], t[1], color="#f2f2ef", zorder=0)
        elif t is not None:
            ax.axhline(t, color="#b8b8b4", lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.axhline(5, color="#c9c9c5", lw=0.8, ls=(0, (2, 3)), zorder=0)
        ax.plot(xs, at5, color=C_AT5, lw=1.8, ls="--", marker="o", ms=5, zorder=2,
                label="refusals scored at 5")
        ax.plot(xs, gen, color=C_GENUINE, lw=2.0, marker="o", ms=6, zorder=3,
                label="genuine answers")
        ax.plot([x_cold], [r["cold"]["at5"]], color=C_AT5, marker="o", ms=5, ls="none", zorder=2)
        ax.plot([x_cold], [r["cold"]["genuine"]], color=C_GENUINE, marker="o", ms=6, ls="none", zorder=3)
        ax.axvline(k - 0.5 + 0.45, color="#d9d9d5", lw=0.8, zorder=0)
        ax.set_xlim(-0.5, x_cold + 0.5)
        ax.set_ylim(-0.3, 10.3)
        ax.set_yticks(range(0, 11, 2))
        ax.set_xticks(list(xs) + [x_cold])
        share_txt = [("" if np.isnan(s) else f"{100*s:.0f}%") for s in sh] + \
                    [("" if np.isnan(r["cold"]["share"]) else f"{100*r['cold']['share']:.0f}%")]
        ax.set_xticklabels([f"{lab}\n{s}" for lab, s in zip(labels + ["no\npersona"], share_txt)],
                           fontsize=8.2, color=C_TEXT)
        ax.grid(axis="y", color=C_GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(NAMES[m], loc="left", fontsize=12, fontweight="bold", color=C_TEXT, pad=8)
        if r["cold"]["n_all"] and all(np.isnan(v) for v in gen):
            ax.text(0.5, 0.62, "no answers took a position", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color=C_MUTED, style="italic")
    for ax in axes[:, 0]:
        ax.set_ylabel(spec["ylabel"], fontsize=9, color=C_MUTED)
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    order = [lbls.index("genuine answers"), lbls.index("refusals scored at 5")]
    fig.legend([handles[i] for i in order], [lbls[i] for i in order], loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, fontsize=10)
    fig.text(0.01, 0.005,
             "Percentages under each identity: share of the cell's answers that took a position; "
             "the rest are refusals or answers on which the judges disagreed. All three turns pooled.",
             fontsize=8.2, color=C_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--topic")
    ap.add_argument("--out")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    scored = pd.read_csv(a.scored, low_memory=False)
    if a.all:
        for t in TOPICS:
            p = draw(cell_stats(scored, t), t, Path(a.outdir) / f"{t}_six_models.png")
            print("wrote", p)
    else:
        p = draw(cell_stats(scored, a.topic), a.topic, a.out)
        print("wrote", p)


if __name__ == "__main__":
    main()
