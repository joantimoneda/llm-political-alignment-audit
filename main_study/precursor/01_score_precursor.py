##
## Created by: Joan Timoneda
## Date: 12 September 2026
##

"""Score the lean of each model's replies to the user's identity statements
(turns 2 to 5), before the direct question is asked. Both judges see the
user's statement and the reply and rate only the lean the reply asserts.
Calls are banked in data/lean_scores.jsonl; the main study's store is never
touched.
"""

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
sys.path.insert(0, str(MAIN))
from config import JUDGE_MODELS, JUDGE_PARAMS, MODELS, TOPICS
from model_client import chat

LONG_DEFAULT = MAIN / "data" / "processed" / "long.csv"
DATA = HERE / "data"
STORE = DATA / "lean_scores.jsonl"
SCORED = DATA / "lean_scored.csv"
CUE_TURNS_DEFAULT = (2, 3, 4, 5)
RUBRICS = {t: HERE / "prompts" / f"lean_{t}.txt" for t in
           ("abortion", "catalan", "climate", "nazism", "control")}

PRICE_IN = {"gpt": 5.0, "claude": 5.0}
PRICE_OUT = {"gpt": 30.0, "claude": 25.0}
OUT_TOKENS_HIGH = 400
TOKENS_PER_WORD = 1.33
EMPIRICAL_PER_CALL = 0.0137


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_prompt(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if not ln.lstrip().startswith("#")]
    return "\n".join(body).strip()


def fill(prompt, user, assistant):
    if "{user}" not in prompt or "{assistant}" not in prompt:
        raise ValueError("lean rubric needs {user} and {assistant} slots")
    return prompt.replace("{user}", str(user)).replace("{assistant}", str(assistant))


def parse_lean(text):
    m = re.fullmatch(r"\s*(10|[0-9])\s*", text or "")
    if m:
        return int(m.group(1))
    hits = re.findall(r"(?<![\d.])(10|[0-9])(?![\d.])", text or "")
    return int(hits[0]) if len(hits) == 1 else None


def judge_once(judge, prompt_text):
    raw = (chat(judge, [{"role": "user", "content": prompt_text}], **JUDGE_PARAMS) or "").strip()
    val = parse_lean(raw)
    if val is None:
        raw = (chat(judge, [{"role": "user", "content":
                             prompt_text + "\n\nReply with only a single integer from 0 to 10."}],
                    **JUDGE_PARAMS) or "").strip()
        val = parse_lean(raw)
    return val, raw


def select_replies(long_df, topics, models=None, cue_turns=CUE_TURNS_DEFAULT):
    d = long_df[(long_df.turn_type == "cue") & long_df.turn_index.isin(list(cue_turns))
                & long_df.topic.isin(topics) & (long_df.cue_level != "cold")].copy()
    if models:
        d = d[d.model.isin(models)]
    d["cue_step"] = d.turn_index - 1
    d = d.sort_values(["topic", "model", "cue_level", "run", "turn_index"]).reset_index(drop=True)
    return d


def load_store(path=STORE):
    done, dropped = {}, 0
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
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


def append_store(rec, path=STORE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def estimate_cost(replies, judges, rubric_words):
    words = (replies.user.fillna("").str.split().str.len()
             + replies.assistant.fillna("").str.split().str.len())
    tokens_in = float(((words + rubric_words) * TOKENS_PER_WORD).sum())
    lines = []
    total_low = total_high = 0.0
    for j in judges:
        low = tokens_in / 1e6 * PRICE_IN[j]
        high = low + len(replies) * OUT_TOKENS_HIGH / 1e6 * PRICE_OUT[j]
        total_low += low
        total_high += high
        lines.append(f"  {j:7s}: {len(replies):6d} calls, ~{tokens_in/1e6:.1f}M input tokens "
                     f"-> ${low:,.0f} (input only) to ${high:,.0f} (with {OUT_TOKENS_HIGH} output tokens/call)")
    lines.append(f"  total  : ${total_low:,.0f} to ${total_high:,.0f}; cross-check at the main battery's "
                 f"empirical rate: ${len(replies) * len(judges) * EMPIRICAL_PER_CALL:,.0f}")
    return "\n".join(lines)


def _score_pair(key, judge, row, prompt_text, sleep):
    val, raw = judge_once(judge, prompt_text)
    rec = {"key": key, "cell_stem": row.cell_stem, "turn_index": int(row.turn_index),
           "judge": judge, "pass": "lean",
           "value": val if val is not None else "PARSE_FAIL", "raw": raw, "ts": _now()}
    if sleep:
        time.sleep(sleep)
    return rec


def run(replies, judges, store, dry_run, limit, sleep, workers=1):
    calls, pending = 0, 0
    rubrics = {t: load_prompt(RUBRICS[t]) for t in replies.topic.unique()}
    if limit:
        replies = replies.sample(frac=1, random_state=20260915)
    todo = []
    for _, row in replies.iterrows():
        for judge in judges:
            key = f"{row.cell_stem}|{row.turn_index}|{judge}|lean"
            if key in store:
                continue
            pending += 1
            if dry_run or (limit and len(todo) >= limit):
                continue
            todo.append((key, judge, row))
    if workers <= 1:
        for key, judge, row in todo:
            rec = _score_pair(key, judge, row, fill(rubrics[row.topic], row.user, row.assistant), sleep)
            append_store(rec)
            store[key] = rec
            calls += 1
        return calls, pending
    lock = threading.Lock()

    def worker(key, judge, row):
        rec = _score_pair(key, judge, row, fill(rubrics[row.topic], row.user, row.assistant), sleep)
        with lock:
            append_store(rec)
            store[key] = rec
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, key, judge, row) for key, judge, row in todo]
        try:
            for f in as_completed(futures):
                f.result()
                calls += 1
        except BaseException:
            ex.shutdown(wait=True, cancel_futures=True)
            raise
    return calls, pending


def merge(replies, judges, store):
    rows = []
    for _, row in replies.iterrows():
        out = {k: row[k] for k in ("cell_stem", "model", "model_id", "topic", "cue_level",
                                   "cue_code", "run", "turn_index", "cue_step", "user", "assistant")}
        vals = []
        for j in judges:
            v = store.get(f"{row.cell_stem}|{row.turn_index}|{j}|lean", {}).get("value")
            v = v if isinstance(v, int) else None
            out[f"lean_{j}"] = v
            out[f"self_judge_{j}"] = (MODELS[j]["developer"] == MODELS[row.model]["developer"]
                                      if row.model in MODELS else False)
            if v is not None:
                vals.append(v)
        out["n_judges"] = len(vals)
        out["lean_mean"] = sum(vals) / len(vals) if vals else float("nan")
        out["lean_spread"] = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
        rows.append(out)
    df = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORED, index=False)
    return df


def summarize(df, judges):
    print(f"\nlean table: {len(df)} replies | {int((df.n_judges == len(judges)).sum())} scored by all judges")
    if len(judges) > 1 and (df.n_judges == len(judges)).any():
        full = df[df.n_judges == len(judges)]
        a, b = full[f"lean_{judges[0]}"], full[f"lean_{judges[1]}"]
        print(f"judge agreement ({judges[0]} vs {judges[1]}): r = {a.corr(b):.3f}, "
              f"exact = {(a == b).mean():.1%}, within 1 = {((a - b).abs() <= 1).mean():.1%}")
    print("\nmean lean by topic x model x cue_step (all judges pooled):")
    print(df.pivot_table(index=["topic", "model"], columns="cue_step", values="lean_mean",
                         aggfunc="mean").round(2).to_string())


def main(a):
    long_df = pd.read_csv(a.long, low_memory=False)
    for t in a.topics:
        if t not in RUBRICS:
            raise SystemExit(f"no lean rubric for topic '{t}' (have: {', '.join(RUBRICS)})")
        if t not in TOPICS:
            raise SystemExit(f"topic '{t}' not in config.TOPICS")
    for j in a.judges:
        if j not in MODELS:
            raise SystemExit(f"unknown judge '{j}' (have: {', '.join(MODELS)})")
    replies = select_replies(long_df, a.topics, a.models, tuple(a.cue_turns))
    if replies.empty:
        raise SystemExit("no pre-question replies selected (check --topics / --models / --cue-turns)")
    store = load_store()
    print(f"selected {len(replies)} pre-question replies: topics={a.topics}, "
          f"cue turns={list(a.cue_turns)}, judges={a.judges}, store has {len(store)} records")
    if a.merge_only:
        df = merge(replies, a.judges, store)
        print(f"merge-only: wrote {SCORED}")
        summarize(df, a.judges)
        return
    calls, pending = run(replies, a.judges, store, a.dry_run, a.limit, a.sleep, a.workers)
    if a.dry_run:
        todo = replies
        rubric_words = max(len(load_prompt(RUBRICS[t]).split()) for t in a.topics)
        print(f"dry run: {pending} judge calls pending ({len(replies)} replies x {len(a.judges)} judges, "
              f"minus banked). Estimated cost for the full selection:")
        print(estimate_cost(todo, a.judges, rubric_words))
        return
    print(f"made {calls} judge calls; {pending - calls} still pending; {len(store)} records banked")
    df = merge(replies, a.judges, store)
    print(f"wrote {SCORED}")
    summarize(df, a.judges)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Lean of the pre-question replies.")
    ap.add_argument("--long", default=str(LONG_DEFAULT), help="long table from 02_build.py")
    ap.add_argument("--topics", nargs="+", default=["abortion"])
    ap.add_argument("--models", nargs="+", default=None, help="subset of systems (default all)")
    ap.add_argument("--cue-turns", nargs="+", type=int, default=list(CUE_TURNS_DEFAULT),
                    help="turn indices to score (2-5 = the four identity statements)")
    ap.add_argument("--judges", nargs="+", default=list(JUDGE_MODELS))
    ap.add_argument("--limit", type=int, default=0, help="cap judge calls this run (pilot)")
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between judge calls")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent judge calls (threads)")
    ap.add_argument("--dry-run", action="store_true", help="count pending calls + cost, no API")
    ap.add_argument("--merge-only", action="store_true", help="no API; rebuild lean_scored.csv")
    main(ap.parse_args())
