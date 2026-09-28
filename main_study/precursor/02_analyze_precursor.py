##
## Created by: Joan Timoneda
## Date: 12 September 2026
##

"""Tables for the pre-question lean scores: cell means by model, identity,
and cue step; the slope of lean on the identity code (OLS, HC1); judge
agreement; and the blind sheet for the human check.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from config import JUDGE_MODELS, TOPICS

SCORED = HERE / "data" / "lean_scored.csv"
RESULTS = HERE / "results"
SIDE_LO, SIDE_HI = 3.5, 6.5


def ols_hc1(y, X):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    meat = (X * (e[:, None] ** 2)).T @ X
    cov = XtX_inv @ meat @ XtX_inv * n / max(n - k, 1)
    return beta, np.sqrt(np.diag(cov))


def p_normal(z):
    return math.erfc(abs(z) / math.sqrt(2))


def slope(df, ycol, step_dummies):
    d = df[[ycol, "cue_code", "cue_step"]].dropna()
    if len(d) < 10 or d.cue_code.nunique() < 2:
        return np.nan, np.nan, np.nan, len(d)
    cols = [np.ones(len(d)), d.cue_code.values]
    if step_dummies:
        for s in sorted(d.cue_step.unique())[1:]:
            cols.append((d.cue_step == s).astype(float).values)
    X = np.column_stack(cols)
    beta, se = ols_hc1(d[ycol].values, X)
    z = beta[1] / se[1] if se[1] > 0 else np.nan
    return beta[1], se[1], (p_normal(z) if np.isfinite(z) else np.nan), len(d)


def cell_means(df, judges):
    rows = []
    for (t, m, lvl, s), g in df.groupby(["topic", "model", "cue_level", "cue_step"]):
        code = g.cue_code.iloc[0]
        out = {"topic": t, "model": m, "cue_level": lvl, "cue_code": code, "cue_step": s,
               "n": len(g), "lean_mean": g.lean_mean.mean()}
        for j in judges:
            out[f"lean_{j}"] = g[f"lean_{j}"].mean()
        sided = (g.lean_mean <= SIDE_LO) | (g.lean_mean >= SIDE_HI)
        out["share_sided"] = sided.mean()
        if pd.notna(code) and code != 0:
            toward = (g.lean_mean - 5) * np.sign(code) > 0
            out["share_toward_user"] = (toward & sided).mean()
            out["share_against_user"] = ((~toward) & sided).mean()
        else:
            out["share_toward_user"] = np.nan
            out["share_against_user"] = np.nan
        rows.append(out)
    cm = pd.DataFrame(rows)
    order = {t: list(TOPICS[t]["cue_code"]) for t in TOPICS}
    cm["_o"] = cm.apply(lambda r: order[r.topic].index(r.cue_level) if r.cue_level in order[r.topic] else 99, axis=1)
    return cm.sort_values(["topic", "model", "_o", "cue_step"]).drop(columns="_o")


def slopes(df, judges):
    rows = []
    for (t, m), g in df.groupby(["topic", "model"]):
        for s in sorted(g.cue_step.unique()):
            b, se, p, n = slope(g[g.cue_step == s], "lean_mean", False)
            rows.append(dict(topic=t, model=m, cue_step=str(int(s)), judge="mean", slope=b, se=se, p=p, n=n))
        b, se, p, n = slope(g, "lean_mean", True)
        rows.append(dict(topic=t, model=m, cue_step="pooled", judge="mean", slope=b, se=se, p=p, n=n))
        for j in judges:
            b, se, p, n = slope(g.rename(columns={f"lean_{j}": "_y"}), "_y", True)
            rows.append(dict(topic=t, model=m, cue_step="pooled", judge=j, slope=b, se=se, p=p, n=n))
    return pd.DataFrame(rows)


def agreement(df, judges):
    if len(judges) < 2:
        return pd.DataFrame()
    a_col, b_col = f"lean_{judges[0]}", f"lean_{judges[1]}"
    rows = []
    for (t, m), g in df.groupby(["topic", "model"]):
        g = g.dropna(subset=[a_col, b_col])
        if len(g) < 5:
            continue
        a, b = g[a_col], g[b_col]
        rows.append(dict(topic=t, model=m, n=len(g), r=a.corr(b), exact=(a == b).mean(),
                         within1=((a - b).abs() <= 1).mean(), mean_abs_diff=(a - b).abs().mean(),
                         disagree_center=(((a == 5) & (b != 5)) | ((a != 5) & (b == 5))).mean()))
    return pd.DataFrame(rows)


def human_sample(df, n, seed, judges):
    rng = np.random.default_rng(seed)
    strata = [g.sample(frac=1, random_state=int(rng.integers(1 << 30)))
              for _, g in df.groupby(["topic", "cue_level", "cue_step"])]
    picked, i = [], 0
    while len(picked) < n and any(len(s) > i for s in strata):
        for s in strata:
            if len(s) > i and len(picked) < n:
                picked.append(s.iloc[i])
        i += 1
    smp = pd.DataFrame(picked).reset_index(drop=True)
    smp = smp.sample(frac=1, random_state=seed).reset_index(drop=True)
    smp["sample_id"] = [f"P{k+1:03d}" for k in range(len(smp))]
    RESULTS.mkdir(parents=True, exist_ok=True)
    blind = smp[["sample_id", "topic", "user", "assistant"]].copy()
    blind["human_lean"] = ""
    blind["notes"] = ""
    blind.to_csv(RESULTS / "human_check_sample.csv", index=False)
    key_cols = ["sample_id", "cell_stem", "turn_index", "model", "cue_level", "cue_step", "lean_mean"] + \
               [f"lean_{j}" for j in judges]
    smp[key_cols].to_csv(RESULTS / "human_check_key.csv", index=False)
    print(f"wrote {RESULTS/'human_check_sample.csv'} ({len(smp)} replies, blind) and human_check_key.csv "
          f"(keep the key closed until the sheet is coded)")


def human_report(path, judges):
    filled = pd.read_csv(path)
    key = pd.read_csv(RESULTS / "human_check_key.csv")
    d = filled.merge(key, on="sample_id")
    d = d[pd.to_numeric(d.human_lean, errors="coerce").notna()].copy()
    d["human_lean"] = d.human_lean.astype(float)
    if d.empty:
        raise SystemExit("no coded rows in the human file (human_lean is empty)")
    h, j = d.human_lean, d.lean_mean
    print(f"human vs two-judge mean, n = {len(d)}: r = {h.corr(j):.3f}, mean |diff| = {(h-j).abs().mean():.2f}, "
          f"within 1 = {((h-j).abs() <= 1).mean():.1%}")
    for jj in judges:
        print(f"human vs {jj}: r = {h.corr(d[f'lean_{jj}']):.3f}, within 1 = {((h - d[f'lean_{jj}']).abs() <= 1).mean():.1%}")
    off = d[(d.lean_mean <= SIDE_LO) | (d.lean_mean >= SIDE_HI)]
    if len(off):
        ack = (off.human_lean.between(4.5, 5.5)).mean()
        print(f"of {len(off)} replies the judges scored as taking a side, the human read {ack:.0%} as "
              f"acknowledgement only (5): this is the agree-vs-acknowledge check")
    ctr = d[d.lean_mean.between(4.5, 5.5)]
    if len(ctr):
        print(f"of {len(ctr)} replies the judges scored at the center, the human saw a side in "
              f"{((ctr.human_lean <= SIDE_LO) | (ctr.human_lean >= SIDE_HI)).mean():.0%}")


def main(a):
    df = pd.read_csv(SCORED, low_memory=False)
    judges = [j for j in JUDGE_MODELS if f"lean_{j}" in df.columns]
    df = df[df.n_judges >= 1].copy()
    if a.human_file:
        human_report(a.human_file, judges)
        return
    if a.human_sample:
        human_sample(df[df.n_judges == len(judges)], a.human_sample, a.seed, judges)
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    cm = cell_means(df, judges)
    cm.to_csv(RESULTS / "precursor_cell_means.csv", index=False)
    sl = slopes(df, judges)
    sl.to_csv(RESULTS / "precursor_slopes.csv", index=False)
    ag = agreement(df, judges)
    if len(ag):
        ag.to_csv(RESULTS / "precursor_agreement.csv", index=False)
    print(f"replies: {len(df)}; wrote precursor_cell_means.csv, precursor_slopes.csv"
          + (", precursor_agreement.csv" if len(ag) else ""))
    print("\nmean lean by identity (rows) and cue step (columns), per topic and system:")
    for (t, m), g in cm.groupby(["topic", "model"], sort=False):
        print(f"\n[{t} / {m}]")
        print(g.pivot(index="cue_level", columns="cue_step", values="lean_mean")
               .reindex([c for c in TOPICS[t]["cue_code"] if c != "cold"]).round(2).to_string())
    print("\npooled slopes of lean on the identity code (two-judge mean; HC1):")
    print(sl[(sl.cue_step == "pooled") & (sl.judge == "mean")]
          .set_index(["topic", "model"])[["slope", "se", "p", "n"]].round(3).to_string())
    if len(ag):
        print("\njudge agreement:")
        print(ag.set_index(["topic", "model"]).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summaries of the pre-question lean scores.")
    ap.add_argument("--human-sample", type=int, default=0, help="write a blind sample of N replies")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--human-file", default=None, help="filled-in human_check_sample.csv")
    main(ap.parse_args())
