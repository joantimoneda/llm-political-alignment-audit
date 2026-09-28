##
## Created by: Joan Timoneda
## Date: 16 September 2026
##

"""Agreement between the hand codes and the two judges on the 700 pilot
answers. Writes pilot_human_vs_judges.md and .csv.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCORED = HERE / "scored.csv"
OUT_CSV = HERE / "pilot_human_vs_judges.csv"
OUT_MD = HERE / "pilot_human_vs_judges.md"
SCOREABLE = ("TAKES_POSITION", "ASSERTS_DEFAULT")
EXCLUDED = ("REFUSES", "DISPUTED")
JUDGE_COLS = [("mean", "stance_mean"), ("gpt", "stance_gpt"), ("claude", "stance_claude")]
SCALE = list(range(11))


def round_half_up(x):
    return np.floor(np.asarray(x, dtype=float) + 0.5).astype(int)


def qwk(a, b, scale=SCALE):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    k = len(scale)
    idx = {v: i for i, v in enumerate(scale)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    n = obs.sum()
    if n == 0:
        return float("nan")
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float)
    denom = (w * exp).sum()
    return float("nan") if denom == 0 else 1.0 - (w * obs).sum() / denom


def agreement(h, s):
    h = np.asarray(h, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(h)
    if n < 2:
        return {"n": n, "pearson_r": float("nan"), "within_1": float("nan"),
                "exact": float("nan"), "qwk": float("nan"), "mean_abs_diff": float("nan")}
    r = float(np.corrcoef(h, s)[0, 1]) if h.std() > 0 and s.std() > 0 else float("nan")
    return {"n": n,
            "pearson_r": r,
            "within_1": float((np.abs(h - s) <= 1).mean()),
            "exact": float((h == s).mean()),
            "qwk": qwk(h.astype(int), round_half_up(s)),
            "mean_abs_diff": float(np.abs(h - s).mean())}


def main():
    df = pd.read_csv(SCORED)
    rows = []

    def add(subset, judge, stat, value, n):
        rows.append({"subset": subset, "judge": judge, "statistic": stat,
                     "value": value, "n": n})

    subsets = [("all", df), ("test", df[df.turn == "test"]), ("pushback", df[df.turn == "pushback"])]
    text = ["# Human vs judge agreement on the 700 hand-coded pilot answers", ""]
    text.append(f"Source: `validation/scored.csv` ({len(df)} answers; {int(df.label_consensus.notna().sum())} "
                f"fully labeled by both judges). Hand code = column `human` (0-10 abortion scale, "
                f"10 = legal in all cases). Judges = GPT-5.5 and Claude Opus 4.8 running the study's "
                f"refusal classifier and abortion stance rubric unchanged.")
    text.append("")
    for name, d in subsets:
        labeled = d[d.label_consensus.notna()]
        n_lab = len(labeled)
        counts = labeled.label_consensus.value_counts().to_dict()
        excl = labeled[labeled.label_consensus.isin(EXCLUDED)]
        n_excl = len(excl)
        share_excl = n_excl / n_lab if n_lab else float("nan")
        excl_h5 = float((excl.human == 5).mean()) if n_excl else float("nan")
        add(name, "consensus", "n_labeled", n_lab, n_lab)
        for lab in ("TAKES_POSITION", "ASSERTS_DEFAULT", "REFUSES", "DISPUTED", "PARSE_FAIL"):
            add(name, "consensus", f"n_{lab}", counts.get(lab, 0), n_lab)
        add(name, "consensus", "n_excluded_refuses_or_disputed", n_excl, n_lab)
        add(name, "consensus", "share_excluded", share_excl, n_lab)
        add(name, "consensus", "share_excluded_human_scored_5", excl_h5, n_excl)
        n_ref = counts.get("REFUSES", 0)
        ref_h5 = float((excl[excl.label_consensus == "REFUSES"].human == 5).mean()) if n_ref else float("nan")
        add(name, "consensus", "share_refuses_human_scored_5", ref_h5, n_ref)
        big = int((labeled.stance_spread >= 3).sum())
        add(name, "consensus", "n_stance_spread_ge_3", big, n_lab)

        text.append(f"## {name} turn(s)" if name != "all" else "## All answers")
        text.append("")
        text.append(f"{n_lab} answers labeled by both judges. Consensus labels: "
                    + ", ".join(f"{lab} {counts.get(lab, 0)}" for lab in
                                ("TAKES_POSITION", "ASSERTS_DEFAULT", "REFUSES", "DISPUTED", "PARSE_FAIL")
                                if counts.get(lab, 0)) + ".")
        if n_lab:
            text.append(f"{n_excl} answers ({share_excl:.1%}) are REFUSES or DISPUTED and so are excluded from "
                        f"the stance scale; the human scored {excl_h5:.1%} of those exactly 5"
                        + (f" ({ref_h5:.1%} of the REFUSES alone)." if n_ref else "."))
            if big:
                text.append(f"{big} answers have a stance spread of 3 or more points between the judges.")
        text.append("")

        genuine = labeled[labeled.label_consensus.isin(SCOREABLE)]
        for judge, col in JUDGE_COLS:
            g = genuine[genuine[col].notna()]
            st = agreement(g.human, g[col])
            for stat in ("n", "pearson_r", "within_1", "exact", "qwk", "mean_abs_diff"):
                add(name, judge, stat, st[stat], st["n"])
            label = {"mean": "two-judge mean", "gpt": "GPT judge alone", "claude": "Claude judge alone"}[judge]
            if st["n"] >= 2:
                text.append(f"- {label} (n = {st['n']}): r = {st['pearson_r']:.3f}, within one point "
                            f"{st['within_1']:.1%}, exact {st['exact']:.1%}, quadratic-weighted kappa "
                            f"{st['qwk']:.3f}, mean absolute difference {st['mean_abs_diff']:.2f}.")
            else:
                text.append(f"- {label}: fewer than 2 scoreable answers.")
        text.append("")

    hv = df.human.value_counts().sort_index()
    text.append("## Notes")
    text.append("")
    text.append("- Hand codes present: " + ", ".join(f"{int(k)} x{int(v)}" for k, v in hv.items())
                + ". The human used only the upper half of the scale, so r and kappa are computed over a "
                  "restricted range.")
    text.append("- Kappa is Cohen's kappa with quadratic weights over the full 0-10 scale; the two-judge "
                "mean is rounded half up before computing it.")
    text.append("- `self_judge_gpt` is True for every row (the pilot answers were written by an older GPT); "
                "informational only.")
    text.append("- Answers labeled REFUSES or DISPUTED by the judges were still hand-scored; the share of "
                "those the human placed at exactly 5 is the check on whether the classifier's exclusions "
                "coincide with the human's neutral point.")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV} ({len(out)} rows) and {OUT_MD}")
    print()
    print("\n".join(text))

    try:
        from sklearn.metrics import cohen_kappa_score
        genuine = df[df.label_consensus.isin(SCOREABLE) & df.stance_mean.notna()]
        if len(genuine) >= 2:
            mine = qwk(genuine.human.astype(int), round_half_up(genuine.stance_mean))
            theirs = cohen_kappa_score(genuine.human.astype(int), round_half_up(genuine.stance_mean),
                                       weights="quadratic", labels=SCALE)
            assert math.isclose(mine, theirs, abs_tol=1e-9), (mine, theirs)
            print(f"\nkappa cross-check vs scikit-learn: ok ({mine:.6f})")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
