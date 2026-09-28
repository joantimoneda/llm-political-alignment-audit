##
## Created by: Joan Timoneda
## Date: 7 July 2026
##

"""Grok 3 (collected March 2026, rescored with the study's judge) against
Grok 4.3 (July 2026) on the abortion battery: refusal rate and mean stance
by cue level. Reads banked scores only; no API calls.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PROC = HERE / "data" / "processed"
RESULTS = HERE / "results"
SCOREABLE = ("TAKES_POSITION", "ASSERTS_DEFAULT")
CUE_ORDER = ["liberal", "mod_liberal", "neutral", "mod_conservative", "conservative"]


def grok3_answers():
    store = {}
    with open(PROC / "scores.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["cell_stem"].startswith("grok3__"):
                store[r["key"]] = r.get("value")
    mg = pd.read_csv(PROC / "long_march_grok.csv")
    ans = mg[mg.is_scoreable].copy()
    ans["label"] = ans.apply(
        lambda r: store.get(f"{r.cell_stem}|{r.turn_index}|gpt|label"), axis=1)
    ans["stance"] = pd.to_numeric(ans.apply(
        lambda r: store.get(f"{r.cell_stem}|{r.turn_index}|gpt|stance"), axis=1),
        errors="coerce")
    ans["generation"] = "grok-3 (2026-03, rescored)"
    return ans


def grok43_answers():
    sc = pd.read_csv(PROC.parent / "processed_gpt_judge" / "scored.csv")
    d = sc[(sc.model == "grok") & (sc.topic == "abortion")].copy()
    d["label"] = d.label_gpt
    d["stance"] = pd.to_numeric(d.stance_gpt, errors="coerce")
    d.loc[~d.label.isin(SCOREABLE), "stance"] = float("nan")
    d["generation"] = "grok-4.3 (2026-07)"
    return d


def per_cue(ans):
    g = (ans[ans.cue_level != "cold"]
         .groupby(["generation", "scoring_topic", "cue_level"])
         .agg(n=("label", "size"),
              refusal=("label", lambda s: (s == "REFUSES").mean()),
              stance_genuine=("stance", "mean"),
              n_stance=("stance", lambda s: s.notna().sum()))
         .reset_index())
    g["pct_answering"] = 1.0 - g.refusal
    g["cue_level"] = pd.Categorical(g.cue_level, CUE_ORDER, ordered=True)
    return g.sort_values(["scoring_topic", "generation", "cue_level"])


def main():
    both = pd.concat([per_cue(grok3_answers()), per_cue(grok43_answers())],
                     ignore_index=True)
    out = RESULTS / "grok_generations_numbers.csv"
    both.round(4).to_csv(out, index=False)
    print(f"wrote {out}\n")
    for st, g in both.groupby("scoring_topic"):
        print(f"=== {st} ===")
        print(g.pivot_table(index="cue_level", columns="generation", observed=True,
                            values=["refusal", "stance_genuine", "pct_answering"])
               .round(2).to_string())
        print()


if __name__ == "__main__":
    main()
