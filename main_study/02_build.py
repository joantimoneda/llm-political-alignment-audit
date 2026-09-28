##
## Created by: Joan Timoneda
## Date: 30 June 2026
##

"""Stage 2. Turn the raw cells into data/processed/long.csv, one row per turn,
with the cue coding, the scoring plan for each answer, and the parsed
profiling rating.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from config import TOPICS

HERE = Path(__file__).parent
RAW_DEFAULT = HERE / "data" / "raw"
OUT_DEFAULT = HERE / "data" / "processed"

SCOREABLE = {"probe", "pressure", "spillover_probe", "spillover_pressure"}


_SCALE_SPANS = [
    re.compile(r"\b[01]\s*(?:\([^)]{0,40}\))?\s*(?:[-–—]?\s*to\s*[-–—]?|[-–—]|through)\s*10\b"
               r"(?:\s*\([^)]{0,40}\))?(?:\s*(?:point\s+)?scale\b)?", re.I),
    re.compile(r"\bbetween\s+[01]\s+and\s+10\b", re.I),
    re.compile(r"\bout\s+of\s+10\b", re.I),
    re.compile(r"[\"“”][^\"“”\d]{0,25}(?:10|[0-9])[^\"“”\d]{0,10}[\"“”]"),
]
_INT_0_10 = re.compile(r"(?<![\d.])(10|[0-9])(?!\.?\d)")
_BOLD_INT = re.compile(r"\*\*\s*(10|[0-9])\s*\*\*")


def _profiled_number(text):
    """Extract the 0-10 rating the model gave the user on the profiling turn.
    Mentions of the scale itself ("on a 0-10 scale") are removed first so
    their digits are not read as the rating. Returns NaN if no single
    number remains.
    """
    if not isinstance(text, str) or not text:
        return float("nan")
    m = re.fullmatch(r"\W*(10|[0-9])\W{0,3}", text.strip())
    if m:
        return float(m.group(1))
    for pat in _SCALE_SPANS:
        text = pat.sub(" ", text)
    bold = {v for v in _BOLD_INT.findall(text)}
    if len(bold) == 1:
        return float(bold.pop())
    vals = {v for v in _INT_0_10.findall(text)}
    return float(vals.pop()) if len(vals) == 1 else float("nan")


def _norm_pressure_step(turn_type, step):
    if turn_type in ("probe", "spillover_probe"):
        return 0.0
    if turn_type in ("pressure", "spillover_pressure"):
        return float(step) if step is not None else float("nan")
    return float("nan")


def _rows_for_cell(jsonl_path):
    stem = jsonl_path.stem
    meta_file = jsonl_path.parent / f"{stem}.meta.json"
    persona_sha = None
    if meta_file.exists():
        persona_sha = json.loads(meta_file.read_text(encoding="utf-8")).get("persona_sha256_12")

    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        topic, ttype = r["topic"], r.get("turn_type", "turn")
        spec = TOPICS.get(topic)
        if spec is None:
            raise KeyError(f"{stem}: topic '{topic}' not in config.TOPICS")
        scoring_topic = r.get("spillover_topic") if ttype.startswith("spillover") else topic
        if ttype.startswith("spillover"):
            scorer_path = spec["spillover"].get(scoring_topic)
        else:
            scorer_path = spec["scorer"]
        cue_code = spec["cue_code"].get(r["cue_level"])
        rows.append({
            "cell_stem": stem,
            "model": r["model"], "model_id": r.get("model_id"),
            "provider": r.get("provider"),
            "topic": topic, "cue_level": r["cue_level"],
            "cue_code": float(cue_code) if cue_code is not None else float("nan"),
            "run": r["run"],
            "target_type": spec["target_type"], "anchor_clarity": spec["anchor_clarity"],
            "persona_sha12": persona_sha,
            "turn_index": r["turn_index"], "turn_type": ttype,
            "pressure_step": _norm_pressure_step(ttype, r.get("pressure_step")),
            "is_scoreable": ttype in SCOREABLE,
            "scoring_topic": scoring_topic if ttype in SCOREABLE else None,
            "scorer_path": scorer_path if ttype in SCOREABLE else None,
            "profiled_user": _profiled_number(r["assistant"]) if ttype == "profiling" else float("nan"),
            "ts": r.get("ts"),
            "user": r["user"], "assistant": r["assistant"],
        })
    prof = [i for i, r in enumerate(rows) if r["turn_type"] == "profiling"]
    for i in prof[:-1]:
        rows[i]["profiled_user"] = float("nan")
    return rows


def build(raw_dir, out_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    files = sorted(raw_dir.glob("*.jsonl"))
    failed = sorted(raw_dir.glob("*.FAILED.json"))
    if not files:
        raise SystemExit(f"no *.jsonl cells in {raw_dir}; run 01_collect.py first")

    rows = []
    skipped_partial = 0
    for f in files:
        if not (f.parent / f"{f.stem}.meta.json").exists():
            skipped_partial += 1
            continue
        rows.extend(_rows_for_cell(f))
    if skipped_partial:
        print(f"note: skipped {skipped_partial} cell(s) still being written")
    df = pd.DataFrame(rows).sort_values(
        ["model", "topic", "cue_level", "run", "turn_index"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "long.csv"
    df.to_csv(out, index=False)

    cells = df.drop_duplicates("cell_stem")
    print(f"built {out}  ({len(df)} turns, {len(cells)} cells, "
          f"{df.is_scoreable.sum()} scoreable answers)")
    rep = (cells.groupby(["model", "topic"])
                .agg(cues=("cue_level", lambda s: len(set(s))),
                     runs=("cell_stem", "size"))
                .reset_index())
    rep["cues_expected"] = rep["topic"].map(lambda t: len(TOPICS[t]["cue_code"]))
    print("\ncells per model x topic (cues found vs expected, total runs):")
    print(rep.to_string(index=False))
    incomplete = rep[rep.cues < rep.cues_expected]
    if len(incomplete):
        print(f"\nWARNING: {len(incomplete)} model x topic blocks are missing cue cells")
    if failed:
        print(f"\nWARNING: {len(failed)} FAILED cells awaiting re-collection:")
        for f in failed:
            print("  ", f.name)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 2: raw transcripts to long.csv.")
    ap.add_argument("--raw", default=str(RAW_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    a = ap.parse_args()
    build(a.raw, a.out)
