##
## Created by: Joan Timoneda
## Date: 1 July 2026
##

"""Stage 3. Two passes over every scoreable answer in long.csv: a refusal
classifier (TAKES_POSITION, REFUSES, ASSERTS_DEFAULT) and, for answers both
judges agree are scoreable, the topic's 0-10 stance rubric. Every judge call
is banked in scores.jsonl, so the script can be stopped and resumed. Lines
starting with # in a prompt file are notes and are not sent to the judge.
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import JUDGE_MODELS, JUDGE_PARAMS, MODELS
from model_client import chat

HERE = Path(__file__).parent
PROC = HERE / "data" / "processed"
LABELS = ("TAKES_POSITION", "REFUSES", "ASSERTS_DEFAULT")
SCOREABLE_LABELS = ("TAKES_POSITION", "ASSERTS_DEFAULT")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_prompt(rel_path):
    lines = (HERE / rel_path).read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if not ln.lstrip().startswith("#")]
    return "\n".join(body).strip()


def fill(prompt, answer):
    if "{answer}" not in prompt:
        raise ValueError("scoring prompt has no {answer} slot")
    return prompt.replace("{answer}", answer)


def _ask(judge, prompt_text):
    reply = chat(judge, [{"role": "user", "content": prompt_text}], **JUDGE_PARAMS)
    return (reply or "").strip()


def parse_label(text):
    hits = [lab for lab in LABELS if lab in text.upper()]
    return hits[0] if len(hits) == 1 else None


def parse_stance(text):
    m = re.fullmatch(r"\s*(10|[0-9])\s*", text)
    if m:
        return int(m.group(1))
    hits = re.findall(r"(?<![\d.])(10|[0-9])(?![\d.])", text)
    return int(hits[0]) if len(hits) == 1 else None


def judge_once(judge, prompt_text, parser, nudge):
    raw = _ask(judge, prompt_text)
    val = parser(raw)
    if val is None:
        raw = _ask(judge, prompt_text + "\n\n" + nudge)
        val = parser(raw)
    return val, raw


def _store_path(proc_dir):
    return Path(proc_dir) / "scores.jsonl"


def load_store(proc_dir):
    path = _store_path(proc_dir)
    done = {}
    dropped = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            done[r["key"]] = r
    if dropped:
        print(f"note: dropped {dropped} truncated store line(s); those calls re-run")
    return done


def append_store(proc_dir, rec):
    with open(_store_path(proc_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def run_labels(answers, judges, store, proc_dir, dry_run, limit, sleep):
    calls = 0
    clf = load_prompt("prompts/scoring/refusal_classifier.txt")
    for _, row in answers.iterrows():
        for judge in judges:
            key = f"{row.cell_stem}|{row.turn_index}|{judge}|label"
            if key in store:
                continue
            if dry_run or (limit and calls >= limit):
                calls += 1
                continue
            val, raw = judge_once(judge, fill(clf, row.assistant), parse_label,
                                  "Reply with only one label: TAKES_POSITION, REFUSES, or ASSERTS_DEFAULT.")
            rec = {"key": key, "cell_stem": row.cell_stem, "turn_index": int(row.turn_index),
                   "judge": judge, "pass": "label",
                   "value": val if val else "PARSE_FAIL", "raw": raw, "ts": _now()}
            append_store(proc_dir, rec)
            store[key] = rec
            calls += 1
            if sleep:
                time.sleep(sleep)
    return calls


def consensus_label(row_labels):
    vals = [v for v in row_labels if v in LABELS]
    if not vals:
        return "PARSE_FAIL"
    return vals[0] if len(set(vals)) == 1 and len(vals) == len(row_labels) else "DISPUTED"


def run_stances(answers, judges, store, proc_dir, dry_run, limit, sleep):
    calls = 0
    prompt_cache = {}
    for _, row in answers.iterrows():
        labels = [store.get(f"{row.cell_stem}|{row.turn_index}|{j}|label", {}).get("value")
                  for j in judges]
        if None in labels and not dry_run:
            continue
        if not dry_run and consensus_label(labels) not in SCOREABLE_LABELS:
            continue
        for judge in judges:
            key = f"{row.cell_stem}|{row.turn_index}|{judge}|stance"
            if key in store:
                continue
            if dry_run or (limit and calls >= limit):
                calls += 1
                continue
            if row.scorer_path not in prompt_cache:
                prompt_cache[row.scorer_path] = load_prompt(row.scorer_path)
            val, raw = judge_once(judge, fill(prompt_cache[row.scorer_path], row.assistant),
                                  parse_stance,
                                  "Reply with only a single integer from 0 to 10.")
            rec = {"key": key, "cell_stem": row.cell_stem, "turn_index": int(row.turn_index),
                   "judge": judge, "pass": "stance",
                   "value": val if val is not None else "PARSE_FAIL", "raw": raw, "ts": _now()}
            append_store(proc_dir, rec)
            store[key] = rec
            calls += 1
            if sleep:
                time.sleep(sleep)
    return calls


def merge(answers, judges, store, proc_dir):
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
        out["label_consensus"] = consensus_label(labels) if len(labels) == len(judges) else None
        out["genuine"] = out["label_consensus"] in SCOREABLE_LABELS
        out["stance_mean"] = sum(stances) / len(stances) if stances else float("nan")
        out["stance_spread"] = (max(stances) - min(stances)) if len(stances) > 1 else float("nan")
        rows.append(out)
    df = pd.DataFrame(rows)
    out_path = Path(proc_dir) / "scored.csv"
    df.to_csv(out_path, index=False)
    return df, out_path


def summarize(df, judges):
    n = len(df)
    labeled = df.label_consensus.notna().sum()
    print(f"\nscored table: {n} scoreable answers | {labeled} fully labeled")
    if labeled:
        print("\nconsensus labels:")
        print(df.label_consensus.value_counts(dropna=False).to_string())
        disp = (df.label_consensus == "DISPUTED").mean()
        print(f"\njudge disagreement rate (labels): {disp:.1%}")
        big = (df.stance_spread >= 3).sum()
        if big:
            print(f"WARNING: {big} answers with stance spread >= 3 points across judges")


def main(long_path, proc_dir, judges, dry_run, limit, sleep, only_model=None, only_topic=None,
         merge_only=False):
    would_merge = merge_only or not (only_model or only_topic)
    if (set(judges) != set(JUDGE_MODELS) and would_merge and not dry_run
            and Path(proc_dir).resolve() == PROC.resolve()):
        raise SystemExit(
            f"refusing to write {PROC/'scored.csv'} under judges={judges}; that would "
            f"overwrite the frozen two-judge merge (JUDGE_MODELS={JUDGE_MODELS}). "
            f"Pass --proc <other dir>.")
    long_df = pd.read_csv(long_path)
    answers = long_df[long_df.is_scoreable].copy()
    if only_model:
        answers = answers[answers.model == only_model]
    if only_topic:
        answers = answers[answers.topic == only_topic]
    if answers.empty:
        raise SystemExit("no scoreable answers in the long table (after filters)")
    for j in judges:
        if j not in MODELS:
            raise SystemExit(f"unknown judge '{j}' (have: {', '.join(MODELS)})")
    store = load_store(proc_dir)
    if merge_only:
        df, out_path = merge(answers, judges, store, proc_dir)
        print(f"merge-only: wrote {out_path}")
        summarize(df, judges)
        return
    n1 = run_labels(answers, judges, store, proc_dir, dry_run, limit, sleep)
    n2 = run_stances(answers, judges, store, proc_dir, dry_run, limit, sleep)
    if dry_run:
        print(f"dry run: {n1} label calls + >= {n2} stance calls pending "
              f"({len(answers)} answers x {len(judges)} judges)")
        return
    print(f"made {n1} label calls, {n2} stance calls "
          f"({len(store)} total records in scores.jsonl)")
    if only_model or only_topic:
        print("filtered run: skipping scored.csv merge (run unfiltered to merge)")
        return
    df, out_path = merge(answers, judges, store, proc_dir)
    print(f"wrote {out_path}")
    summarize(df, judges)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 3: refusal labels and stance scores.")
    ap.add_argument("--long", default=str(PROC / "long.csv"), help="long table from 02_build.py")
    ap.add_argument("--proc", default=str(PROC), help="output dir (scores.jsonl, scored.csv)")
    ap.add_argument("--judges", nargs="+", default=list(JUDGE_MODELS))
    ap.add_argument("--limit", type=int, default=0, help="cap judge calls this run (pilot)")
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between judge calls")
    ap.add_argument("--dry-run", action="store_true", help="count pending calls, no API")
    ap.add_argument("--only-model", default=None, help="score only this model's answers")
    ap.add_argument("--only-topic", default=None, help="score only this topic's answers")
    ap.add_argument("--merge-only", action="store_true",
                    help="no API calls; just rebuild scored.csv from the store")
    a = ap.parse_args()
    main(a.long, a.proc, a.judges, a.dry_run, a.limit, a.sleep,
         only_model=a.only_model, only_topic=a.only_topic, merge_only=a.merge_only)
