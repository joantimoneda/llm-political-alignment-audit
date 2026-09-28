##
## Created by: Joan Timoneda
## Date: 7 July 2026
##

"""Vaccines appendix. The vaccines battery was collected after the main run
and scored by the GPT judge only. This rebuilds vaccines_scored.csv and
results/vaccines_signatures_full.csv from the banked scores; no API calls.
"""

import importlib.util
from pathlib import Path

import pandas as pd

from config import MODELS, TOPICS

HERE = Path(__file__).parent
PROC = HERE / "data" / "processed"
RESULTS = HERE / "results"
JUDGES = ["gpt"]
TOPIC = "vaccines"


def _load(module_file):
    spec = importlib.util.spec_from_file_location(module_file.stem, module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = _load(HERE / "03_score.py")
analyze = _load(HERE / "04_analyze.py")


def merge_from_store(answers, judges, store):
    rows = []
    for _, row in answers.iterrows():
        out = row.to_dict()
        labels, stances = [], []
        for j in judges:
            lab = store.get(f"{row.cell_stem}|{row.turn_index}|{j}|label", {}).get("value")
            st = store.get(f"{row.cell_stem}|{row.turn_index}|{j}|stance", {}).get("value")
            st = st if isinstance(st, int) else None
            out[f"label_{j}"] = lab
            out[f"stance_{j}"] = st
            out[f"self_judge_{j}"] = (MODELS[j]["developer"] == MODELS[row.model]["developer"]) if row.model in MODELS else False
            if lab:
                labels.append(lab)
            if st is not None:
                stances.append(st)
        out["label_consensus"] = score.consensus_label(labels) if len(labels) == len(judges) else None
        out["genuine"] = out["label_consensus"] in score.SCOREABLE_LABELS
        out["stance_mean"] = sum(stances) / len(stances) if stances else float("nan")
        out["stance_spread"] = (max(stances) - min(stances)) if len(stances) > 1 else float("nan")
        rows.append(out)
    return pd.DataFrame(rows)


def main():
    long_df = pd.read_csv(PROC / "long.csv")
    vac_long = long_df[long_df.topic == TOPIC]
    answers = vac_long[vac_long.is_scoreable].copy()
    if answers.empty:
        raise SystemExit("no vaccines answers in long.csv")

    store = score.load_store(PROC)
    df = merge_from_store(answers, JUDGES, store)
    unlabeled = df.label_consensus.isna().sum()
    if unlabeled:
        print(f"WARNING: {unlabeled} vaccines answers have no store record")
    out_scored = PROC / "vaccines_scored.csv"
    df.to_csv(out_scored, index=False)
    print(f"wrote {out_scored}  ({len(df)} answers)")

    rows = []
    for model, d in df.groupby("model"):
        ld = vac_long[vac_long.model == model]
        out = {"model": model, "topic": TOPIC, "target_type": TOPICS[TOPIC]["target_type"]}
        out.update(analyze.signature(d, ld, TOPIC))
        rows.append(out)
    sig = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    out_sig = RESULTS / "vaccines_signatures_full.csv"
    sig.to_csv(out_sig, index=False)
    print(f"wrote {out_sig}  ({len(sig)} model signatures)")

    show = ["model", "position_cold", "refusal_cold", "refusal_rate", "mirroring",
            "mirroring_p", "steadfastness", "dev_cold", "anchor_drift",
            "position_cold_transmitted", "mirroring_transmitted"]
    print(f"\n=== vaccines ({TOPICS[TOPIC]['target_type']}; scale: {TOPICS[TOPIC]['scale']}) ===")
    print(sig[show].round(2).to_string(index=False))

    canon_path = RESULTS / "vaccines_signatures.csv"
    if canon_path.exists():
        canon = pd.read_csv(canon_path)
        lab = df[df.label_consensus.isin(["TAKES_POSITION", "REFUSES", "ASSERTS_DEFAULT"])].copy()
        lab["stance_t"] = lab.stance_mean.where(lab.label_consensus != "REFUSES", 5.0)
        allcold = (lab[lab.cue_code.isna()].groupby("model").stance_t.mean()
                   .rename("position_allcoldturns"))
        cmp = (canon.merge(sig[["model", "position_cold_transmitted",
                                "mirroring_transmitted"]],
                           on="model", suffixes=("_canon", "_prereg"))
                    .merge(allcold, on="model"))
        print("\ncanonical vs recomputed:")
        print(cmp[["model", "position_cold_transmitted_canon", "position_allcoldturns",
                   "position_cold_transmitted_prereg"]].round(3).to_string(index=False))
        d_all = (cmp.position_cold_transmitted_canon - cmp.position_allcoldturns).abs().max()
        d_mir = (cmp.mirroring_transmitted_canon - cmp.mirroring_transmitted_prereg).abs().max()
        print(f"canonical == all-cold-turns variant: max diff {d_all:.3f} "
              f"({'as expected' if d_all < 0.02 else 'unexpected'})")
        print(f"mirroring_transmitted canonical vs prereg: max diff {d_mir:.3f}")


if __name__ == "__main__":
    main()
