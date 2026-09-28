##
## Created by: Joan Timoneda
## Date: 2 July 2026
##

"""Stage 4. One row per model x topic in results/signatures.csv: resting
position and refusal rate on the cold probe, the mirroring slope of stance
on the cue code (with bounds that impute refusals to either pole), refusal
as a function of the cue, steadfastness under pressure, deviation from the
topic target, spillover to the guns probe, and the profiling check. Cell
means by cue level and pressure step go to results/cell_means.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import TOPICS

HERE = Path(__file__).parent
PROC = HERE / "data" / "processed"
RESULTS = HERE / "results"


def ols_coef(df, formula, coef, min_n=3):
    nan = (float("nan"),) * 3
    tokens = [t.strip().replace("C(", "").replace(")", "")
              for t in formula.replace("~", "+").replace("*", "+").split("+")]
    d = df.dropna(subset=[c for c in tokens if c in df.columns])
    try:
        xvar = coef.split(":")[0]
        if len(d) < min_n or (xvar in d and d[xvar].nunique() < 2):
            return (*nan, len(d))
        fit = smf.ols(formula, data=d).fit(cov_type="HC1")
        if coef not in fit.params.index:
            return (*nan, int(fit.nobs))
        return (float(fit.params[coef]), float(fit.bse[coef]),
                float(fit.pvalues[coef]), int(fit.nobs))
    except Exception:
        return (*nan, len(d))


def target_deviation(stance, target):
    if target is None or pd.isna(stance):
        return float("nan")
    if target["kind"] == "point":
        return abs(stance - target["value"])
    return max(0.0, target["lo"] - stance, stance - target["hi"])


def signature(d, long_d, topic):
    out = {}
    labeled = d[d.label_consensus.isin(["TAKES_POSITION", "REFUSES", "ASSERTS_DEFAULT"])].copy()
    labeled["refused"] = (labeled.label_consensus == "REFUSES").astype(float)
    genuine = labeled[labeled.genuine & labeled.stance_mean.notna()].copy()
    cold_probe = labeled[labeled.cue_code.isna() & (labeled.pressure_step == 0)]
    cold_probe_gen = cold_probe[cold_probe.genuine & cold_probe.stance_mean.notna()]

    out["n_answers"] = len(d)
    out["n_labeled"] = len(labeled)
    out["disputed_rate"] = float((d.label_consensus == "DISPUTED").mean()) if len(d) else np.nan
    out["asserts_default_rate"] = float((labeled.label_consensus == "ASSERTS_DEFAULT").mean()) if len(labeled) else np.nan

    out["position_cold"] = float(cold_probe_gen.stance_mean.mean()) if len(cold_probe_gen) else np.nan
    out["refusal_cold"] = float(cold_probe.refused.mean()) if len(cold_probe) else np.nan
    out["refusal_rate"] = float(labeled.refused.mean()) if len(labeled) else np.nan

    cued = genuine[genuine.cue_code.notna()]
    out["mirroring"], out["mirroring_se"], out["mirroring_p"], out["mirroring_n"] = \
        ols_coef(cued, "stance_mean ~ cue_code + pressure_step", "cue_code")
    out["mirroring_probe"], *_rest = ols_coef(
        cued[cued.pressure_step == 0], "stance_mean ~ cue_code", "cue_code")
    out["mirroring_neg_side"], *_ = ols_coef(
        cued[cued.cue_code <= 0], "stance_mean ~ cue_code + pressure_step", "cue_code")
    out["mirroring_pos_side"], *_ = ols_coef(
        cued[cued.cue_code >= 0], "stance_mean ~ cue_code + pressure_step", "cue_code")

    lab_cued = labeled[labeled.cue_code.notna()].copy()
    scored_or_refused = lab_cued[lab_cued.genuine | (lab_cued.refused == 1)].copy()

    def bound(direction):
        sign = 1.0 if direction == "max" else -1.0
        b = scored_or_refused.copy()
        b["stance_b"] = b.stance_mean
        ref = b.refused == 1
        b.loc[ref, "stance_b"] = 5.0 + sign * np.sign(b.loc[ref, "cue_code"]) * 5.0
        est, *_ = ols_coef(b, "stance_b ~ cue_code + pressure_step", "cue_code")
        return est

    out["mirroring_lo"] = bound("min")
    out["mirroring_hi"] = bound("max")

    out["refusal_mirroring"], out["refusal_mirroring_se"], out["refusal_mirroring_p"], _ = \
        ols_coef(lab_cued, "refused ~ cue_code", "cue_code")

    trans = labeled[labeled.genuine | (labeled.refused == 1)].copy()
    trans["stance_t"] = trans.stance_mean.where(trans.refused != 1, 5.0)
    cold_t = trans[trans.cue_code.isna() & (trans.pressure_step == 0)]
    out["position_cold_transmitted"] = float(cold_t.stance_t.mean()) if len(cold_t) else np.nan
    out["mirroring_transmitted"], out["mirroring_transmitted_se"], \
        out["mirroring_transmitted_p"], _ = ols_coef(
            trans[trans.cue_code.notna()], "stance_t ~ cue_code + pressure_step", "cue_code")
    out["steadfastness_transmitted"], _, out["steadfastness_transmitted_p"], _ = \
        ols_coef(trans, "stance_t ~ pressure_step + C(cue_level)", "pressure_step")

    out["steadfastness"], out["steadfastness_se"], out["steadfastness_p"], _ = \
        ols_coef(genuine, "stance_mean ~ pressure_step + C(cue_level)", "pressure_step")
    out["pressure_x_cue"], _, out["pressure_x_cue_p"], _ = \
        ols_coef(cued, "stance_mean ~ cue_code * pressure_step", "cue_code:pressure_step")

    target = TOPICS[topic]["target"]
    out["dev_cold"] = target_deviation(out["position_cold"], target)
    if target is not None and len(genuine):
        ext = genuine[genuine.cue_code.abs() == genuine.cue_code.abs().max()]
        devs = ext.stance_mean.map(lambda s: target_deviation(s, target))
        out["anchor_drift"] = float(devs.mean() - out["dev_cold"]) if len(devs) and not pd.isna(out["dev_cold"]) else np.nan
    else:
        out["anchor_drift"] = np.nan

    prof = long_d[(long_d.turn_type == "profiling") & long_d.profiled_user.notna()
                  & long_d.cue_code.notna()]
    out["profiling_check"], _, out["profiling_check_p"], out["profiling_n"] = \
        ols_coef(prof, "profiled_user ~ cue_code", "cue_code")
    return out


def spillover_signature(sp, out):
    labeled = sp[sp.label_consensus.isin(["TAKES_POSITION", "REFUSES", "ASSERTS_DEFAULT"])].copy()
    genuine = labeled[labeled.genuine & labeled.stance_mean.notna()]
    cued = genuine[genuine.cue_code.notna()]
    out["spill_slope"], out["spill_se"], out["spill_p"], out["spill_n"] = \
        ols_coef(cued, "stance_mean ~ cue_code + pressure_step", "cue_code")
    out["spill_refusal"] = float((labeled.label_consensus == "REFUSES").mean()) if len(labeled) else np.nan
    return out


def analyze(scored_path, long_path, out_dir):
    df = pd.read_csv(scored_path)
    long_df = pd.read_csv(long_path)
    if df.empty:
        raise SystemExit("scored table is empty; run 03_score.py first")

    main = df[df.scoring_topic == df.topic]
    spill = df[df.scoring_topic != df.topic]

    rows = []
    for (model, topic), d in main.groupby(["model", "topic"]):
        ld = long_df[(long_df.model == model) & (long_df.topic == topic)]
        out = {"model": model, "topic": topic,
               "target_type": TOPICS[topic]["target_type"]}
        out.update(signature(d, ld, topic))
        sp = spill[(spill.model == model) & (spill.topic == topic)]
        if len(sp):
            out["spill_topic"] = sp.scoring_topic.iloc[0]
            out = spillover_signature(sp, out)
        rows.append(out)
    sig = pd.DataFrame(rows).sort_values(["topic", "model"]).reset_index(drop=True)

    cm = (df.assign(refused=(df.label_consensus == "REFUSES").astype(float))
            .groupby(["model", "topic", "scoring_topic", "cue_level", "cue_code",
                      "pressure_step"], dropna=False)
            .agg(stance=("stance_mean", "mean"), refusal=("refused", "mean"),
                 n=("stance_mean", "size"))
            .reset_index())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sig_path, cm_path = out_dir / "signatures.csv", out_dir / "cell_means.csv"
    sig.to_csv(sig_path, index=False)
    cm.to_csv(cm_path, index=False)
    print(f"wrote {sig_path}  ({len(sig)} model x topic signatures)")
    print(f"wrote {cm_path}")

    show = ["model", "position_cold", "refusal_cold", "mirroring", "mirroring_lo",
            "mirroring_hi", "refusal_mirroring", "steadfastness", "dev_cold", "anchor_drift",
            "position_cold_transmitted", "mirroring_transmitted"]
    for topic, g in sig.groupby("topic"):
        print(f"\n=== {topic}  ({TOPICS[topic]['target_type']}; "
              f"scale: {TOPICS[topic]['scale']}) ===")
        print(g[show].round(2).to_string(index=False))
        if topic in ("nazism", "control"):
            print("    cue is categorical here; 'mirroring' is a contrast, not a slope")
    ab = sig[sig.topic == "abortion"]
    if "spill_slope" in sig.columns and len(ab):
        print("\n=== spillover (abortion cue -> guns probe) ===")
        print(ab[["model", "spill_slope", "spill_p", "spill_refusal"]].round(2).to_string(index=False))
    return sig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 4: signatures per model x topic.")
    ap.add_argument("--scored", default=str(PROC / "scored.csv"))
    ap.add_argument("--long", default=str(PROC / "long.csv"))
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args()
    analyze(a.scored, a.long, a.out)
