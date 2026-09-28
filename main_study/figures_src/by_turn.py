##
## Created by: Joan Timoneda
## Date: 16 September 2026
##

"""Stance by conversation turn (probe and the pressure turns) for each model."""

import argparse
from pathlib import Path

import pandas as pd

import abortion_july_style as base

TURN_NAMES = {0: "test question", 1: "first press", 2: "second press"}
TOPICS = ["abortion", "catalan", "climate", "nazism", "control", "vaccines"]


def draw_turn(scored, topic, step, outdir):
    cfg = dict(base.CONFIG[topic])
    cfg["subtitles"] = False
    stem = cfg["title"].split(" \u00b7 6 developers")[0]
    tail = " \u00b7 single judge" if topic == "vaccines" else ""
    cfg["title"] = f"{stem} \u00b7 {TURN_NAMES[step]} \u00b7 n=50/cell{tail}"
    d = scored[scored.pressure_step == step]
    saved = base.CONFIG[topic]
    base.CONFIG[topic] = cfg
    try:
        disp = None
        dd = d[d.scoring_topic == topic]
        if topic != "vaccines":
            disp = 100 * (~dd.label_consensus.isin(["TAKES_POSITION", "REFUSES", "ASSERTS_DEFAULT"])).mean()
        out = Path(outdir) / f"{topic}_six_models_turn{step}.png"
        base.draw(base.stats(d, topic), out, topic, disputed_pct=disp)
    finally:
        base.CONFIG[topic] = saved
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--vaccines", default=None)
    ap.add_argument("--outdir", default="results/by_turn")
    a = ap.parse_args()
    scored = pd.read_csv(a.scored, low_memory=False)
    vac = pd.read_csv(a.vaccines, low_memory=False) if a.vaccines else None
    for topic in TOPICS:
        src = vac if topic == "vaccines" else scored
        if src is None:
            continue
        for step in (0, 1, 2):
            print("wrote", draw_turn(src, topic, step, a.outdir))


if __name__ == "__main__":
    main()
