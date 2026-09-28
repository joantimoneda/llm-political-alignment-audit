##
## Created by: Joan Timoneda
## Date: 19 September 2026
##

"""Stance by user identity for each model, one figure per topic (the paper's main figures)."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["gpt", "claude", "grok", "deepseek", "mistral", "gemma"]
CONFIG = {
    "abortion": dict(levels=["liberal", "mod_liberal", "neutral", "mod_conservative", "conservative"],
                     labels=["lib", "m-lib", "neu", "m-con", "con"],
                     ylabel="stance (0=illegal \u2026 10=legal)",
                     title="Abortion \u2014 stance by stated user ideology \u00b7 6 developers \u00b7 n=50 runs/cue (~150 answers/cell)",
                     subtitles=True, band=True),
    "control": dict(levels=["anti", "neutral", "pro"],
                    labels=["anti", "neutral", "pro"],
                    ylabel="stance (0=anti \u2026 10=pro pineapple pizza)",
                    title="Control (pineapple on pizza) \u2014 stance by stated user preference \u00b7 6 developers \u00b7 n=50 runs/cue",
                    subtitles=False, band=False),
    "catalan": dict(levels=["union", "mod_union", "neutral", "mod_independence", "independence"],
                    labels=["union", "m-union", "neutral", "m-indep", "indep"],
                    ylabel="stance (0=unionist \u2026 10=independence)",
                    title="Catalan independence \u2014 stance by stated user identity \u00b7 6 developers \u00b7 n=50 runs/cue (~150 answers/cell)",
                    subtitles=False, band=True),
    "climate": dict(levels=["denier", "mod_skeptic", "neutral", "mod_accept", "accept"],
                    labels=["denier", "m-skeptic", "neutral", "m-accept", "accept"],
                    ylabel="stance (0=hoax \u2026 10=IPCC-consistent)",
                    title="Climate change \u2014 stance by stated user view of the science \u00b7 6 developers \u00b7 n=50 runs/cue (~150 answers/cell)",
                    subtitles=False, band=False, target_note="scale midpoint (5); preregistered target = 10 (top of the scale)"),
    "nazism": dict(levels=["sympathetic", "neutral"],
                   labels=["sympathetic", "neutral"],
                   ylabel="stance (0=exculpates \u2026 10=condemns)",
                   title="Nazism \u2014 stance by stated user identity \u00b7 6 developers \u00b7 n=50 runs/cue (~150 answers/cell)",
                   subtitles=False, band=False, target_note="scale midpoint (5); preregistered target = 10 (top of the scale)",
                   recode=True,
                   foot_extra="One-word answers at the pressure turns (\"Yes.\") are scored at their in-context value, 10 or 0 (Appendix D)."),
    "vaccines": dict(levels=["denier", "mod_skeptic", "neutral", "mod_accept", "accept"],
                     labels=["denier", "m-skeptic", "neutral", "m-accept", "accept"],
                     ylabel="stance (0=anti-vaccine \u2026 10=consensus)",
                     title="Vaccine safety \u2014 stance by stated user view of the evidence \u00b7 6 developers \u00b7 n=50 runs/cue \u00b7 single judge",
                     subtitles=False, band=False, target_note="scale midpoint (5); target = 10 (top of the scale)",
                     foot_override="Blue numbers: % of the cell's answers that took a position; the remainder are refusals. "
                                   "Scored by GPT-5.5 alone (a post-registration appendix arm), so no answer is judge-disputed."),
}
SUBTITLES = {
    "gpt": "answers nearly everything",
    "claude": "refuses — except conservatives",
    "grok": "refuses; rare answers mirror hard",
    "deepseek": "refuses — except the undecided",
    "mistral": "refuses, mild liberal tilt when answering",
    "gemma": "100% refusal — no positions taken",
}

C_GENUINE = "#2a78d6"
C_AT5 = "#eb6834"
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"
C_BAND = "#f0f0ec"


AFFIRM = {"yes", "absolutely", "definitely", "unequivocally", "evil", "it", "undeniably",
          "undoubtedly", "certainly"}


def recode_one_word(df):
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


def stats(scored, topic):
    LEVELS = CONFIG[topic]["levels"]
    d = scored[scored.scoring_topic == topic].copy()
    if CONFIG[topic].get("recode"):
        d = recode_one_word(d)
    lab = d[d.label_consensus.isin(["TAKES_POSITION", "REFUSES", "ASSERTS_DEFAULT"])].copy()
    lab["refused"] = lab.label_consensus.eq("REFUSES")
    lab["at5"] = lab.stance_mean.where(~lab.refused, 5.0)
    gen = lab[lab.genuine & lab.stance_mean.notna()]
    out = {}
    for m in MODELS:
        rows = {}
        for lvl in LEVELS + ["cold"]:
            g = gen[(gen.model == m) & (gen.cue_level == lvl)]
            l = lab[(lab.model == m) & (lab.cue_level == lvl)]
            rows[lvl] = dict(genuine=g.stance_mean.mean() if len(g) else np.nan,
                             at5=l.at5.mean() if len(l) else np.nan,
                             pct=100 * (1 - l.refused.mean()) if len(l) else np.nan)
        out[m] = rows
    return out


def draw(st, out_path, topic="abortion", disputed_pct=None):
    cfg = CONFIG[topic]; LEVELS, LABELS = cfg["levels"], cfg["labels"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4), sharey=True)
    k = len(LEVELS)
    xs = np.arange(k)
    x_cold = k + (1.0 if k >= 5 else 0.15)
    for ax, m in zip(axes.flat, MODELS):
        r = st[m]
        gen = np.array([r[l]["genuine"] for l in LEVELS], dtype=float)
        at5 = np.array([r[l]["at5"] for l in LEVELS], dtype=float)
        if cfg.get("band", True):
            ax.axhspan(4, 6, color=C_BAND, zorder=0)
        ax.axhline(5, color="#b8b8b4", lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.axvline(k - 0.5 + (x_cold - k + 0.5) / 2, color="#e6e6e3", lw=1.0, zorder=0)
        ax.plot(xs, at5, color=C_AT5, lw=2.2, ls="--", marker="o", ms=7, zorder=2)
        ok = ~np.isnan(gen)
        if ok.any():
            ax.plot(xs[ok], gen[ok], color=C_GENUINE, lw=2.2, marker="o", ms=7, zorder=3)
        ax.plot([x_cold], [r["cold"]["at5"]], color=C_AT5, marker="o", ms=7, ls="none", zorder=2)
        if not np.isnan(r["cold"]["genuine"]):
            ax.plot([x_cold], [r["cold"]["genuine"]], color=C_GENUINE, marker="o", ms=7, ls="none", zorder=3)
        if not ok.any():
            ax.text(0.5, 0.72, "no blue line:\nzero answers took a position", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9.5, color=C_MUTED, style="italic")
        ax.set_xlim(-0.5, x_cold + (0.5 if k >= 5 else 0.35))
        ax.set_ylim(-0.5, 10.5)
        ax.set_yticks(range(0, 11, 2))
        ax.set_xticks(list(xs) + [x_cold])
        ax.set_xticklabels(LABELS + ["cold"], fontsize=9.5, color=C_TEXT)
        for x, lvl in zip(list(xs) + [x_cold], LEVELS + ["cold"]):
            p = r[lvl]["pct"]
            ax.text(x, -1.85, "" if np.isnan(p) else f"{p:.0f}", ha="center", va="top",
                    fontsize=9.5, color=C_GENUINE, clip_on=False)
        ax.grid(axis="y", color="#ececea", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#b8b8b4"); ax.spines["bottom"].set_color("#b8b8b4")
        ax.tick_params(length=0)
        ax.set_title(m, loc="left", fontsize=13, fontweight="bold", color=C_TEXT, pad=16 if cfg["subtitles"] else 8)
        if cfg["subtitles"]:
            ax.text(0.0, 1.03, SUBTITLES[m], transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=9.5, color=C_MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel(cfg["ylabel"], fontsize=10, color=C_TEXT)
    fig.suptitle(cfg["title"],
                 x=0.045, ha="left", fontsize=14, fontweight="bold", color=C_TEXT, y=0.985)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C_AT5, lw=2.2, ls="--", marker="o", ms=7),
               Line2D([0], [0], color=C_GENUINE, lw=2.2, marker="o", ms=7),
               Line2D([0], [0], color="#b8b8b4", lw=1.0, ls=(0, (4, 3)))]
    labels = ["Refusals scored at 5 — mean over all answers",
              "Genuine answers only — mean among answers that took a position; blue % below = their share",
              cfg.get("target_note",
                      "scale midpoint (5); shaded band = preregistered neutral range [4, 6]" if cfg.get("band", True)
                      else "scale midpoint (5); no preregistered target on the control")]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.045, 0.955), frameon=False,
               fontsize=10, handlelength=2.6, labelspacing=0.5)
    if disputed_pct is None:
        foot = "Blue numbers: % of the cell's answers that took a position; the remainder are refusals (and ~4% judge-disputed)."
    else:
        foot = ("Blue numbers: % of the cell's labeled answers that took a position; the remainder are refusals. "
                f"Answers on which the judges disagreed ({disputed_pct:.0f}% on this topic) are excluded from both lines.")
    if cfg.get("foot_extra"):
        foot = foot + "\n" + cfg["foot_extra"]
    if cfg.get("foot_override"):
        foot = cfg["foot_override"]
    fig.text(0.045, 0.012, foot, fontsize=9.5, color=C_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035 + (0.02 if cfg.get("foot_extra") else 0), 1, 0.845))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--topic", default="abortion")
    ap.add_argument("--out", default="figures/abortion_six_models_v4.png")
    ap.add_argument("--disputed-footnote", action="store_true",
                    help="print the topic's judge-disputed share in the footnote instead of the abortion wording")
    a = ap.parse_args()
    scored = pd.read_csv(a.scored, low_memory=False)
    disp = None
    if a.disputed_footnote:
        d = scored[scored.scoring_topic == a.topic]
        disp = 100 * (~d.label_consensus.isin(["TAKES_POSITION", "REFUSES", "ASSERTS_DEFAULT"])).mean()
    print("wrote", draw(stats(scored, a.topic), a.out, a.topic, disputed_pct=disp))
