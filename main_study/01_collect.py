##
## Created by: Joan Timoneda
## Date: 29 June 2026
##

"""Stage 1. Run the persona script against each model and save one cell per
model x topic x cue level x run under data/raw/. Resumes by skipping cells
whose files exist; a failed cell is saved as *.FAILED.json and retried on
the next run.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config import ALL, CORE_TOPICS, MODELS, TOPICS
from model_client import chat, resolve_params

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROMPTS = HERE / "prompts"
SYSTEM_PROMPT = "Please interact with the user."
SCHEMA_VERSION = 1


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _persona_path(topic):
    return PROMPTS / "personas" / f"{topic}.txt"


def _parse_persona(path):
    sections, cur = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            cur = s[1:-1].strip()
            sections.setdefault(cur, [])
        elif cur is not None and s and not s.startswith("#"):
            sections[cur].append(s)
    return sections


def _sha12(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def validate_protocol(topic):
    if topic not in TOPICS:
        raise KeyError(f"topic '{topic}' not in config.TOPICS (have: {', '.join(TOPICS)})")
    spec, path = TOPICS[topic], _persona_path(topic)
    if not path.exists():
        raise FileNotFoundError(f"{topic}: persona file missing: {path}")
    persona = _parse_persona(path)

    want = {f"cue {lvl}" for lvl in spec["cue_code"]}
    have = {k for k in persona if k.startswith("cue ")}
    if want - have:
        raise ValueError(f"{topic}: persona file lacks sections {sorted(want - have)} "
                         f"(an empty [cue cold] section must still be present)")
    if have - want:
        raise ValueError(f"{topic}: persona has cue sections not in "
                         f"config.TOPICS[...]['cue_code']: {sorted(have - want)}")
    if len(persona.get("probe", [])) != 1:
        raise ValueError(f"{topic}: [probe] must contain exactly one line")
    if not persona.get("pressure"):
        raise ValueError(f"{topic}: [pressure] must contain at least one line")
    if spec["profiling"] != bool(persona.get("profiling")):
        raise ValueError(f"{topic}: [profiling] presence disagrees with "
                         f"config.TOPICS[...]['profiling']={spec['profiling']}")
    spill_have = {k.split(maxsplit=1)[1] for k in persona
                  if k.startswith("spillover ") and len(k.split()) > 1}
    if set(spec["spillover"]) != spill_have:
        raise ValueError(f"{topic}: spillover sections {sorted(spill_have)} != "
                         f"config {sorted(spec['spillover'])}")
    for rel in [spec["scorer"], *spec["spillover"].values()]:
        if not (HERE / rel).exists():
            raise FileNotFoundError(f"{topic}: scorer prompt missing: {rel}")
    return persona


def build_turns(topic, cue_level, persona=None):
    persona = validate_protocol(topic) if persona is None else persona
    key = f"cue {cue_level}"
    if key not in persona:
        raise ValueError(f"{topic}: no [{key}] section (have: "
                         f"{sorted(k[4:] for k in persona if k.startswith('cue '))})")
    turns = [{"turn_type": "cue", "cue_level": cue_level, "text": t}
             for t in persona[key]]
    turns.append({"turn_type": "probe", "text": persona["probe"][0]})
    turns += [{"turn_type": "pressure", "pressure_step": i + 1, "text": t}
              for i, t in enumerate(persona.get("pressure", []))]
    turns += [{"turn_type": "profiling", "text": t} for t in persona.get("profiling", [])]
    for k, lines in persona.items():
        if k.startswith("spillover") and lines:
            sp = k.split(maxsplit=1)[1] if len(k.split()) > 1 else "spillover"
            turns.append({"turn_type": "spillover_probe", "spillover_topic": sp, "text": lines[0]})
            turns += [{"turn_type": "spillover_pressure", "spillover_topic": sp,
                       "pressure_step": j + 1, "text": t} for j, t in enumerate(lines[1:])]
    return turns


class SessionError(RuntimeError):
    def __init__(self, records, cause):
        super().__init__(str(cause))
        self.records, self.cause = records, cause


def run_session(model_key, turns, sleep=1.0, **chat_kw):
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    records = []
    for i, turn in enumerate(turns):
        history.append({"role": "user", "content": turn["text"]})
        try:
            reply = ""
            for _ in range(3):
                reply = (chat(model_key, history, **chat_kw) or "").strip()
                if reply:
                    break
            if not reply:
                raise RuntimeError("empty reply after 3 attempts")
        except Exception as e:
            raise SessionError(records, e) from e
        history.append({"role": "assistant", "content": reply})
        rec = {k: v for k, v in turn.items() if k != "text"}
        rec.update({"turn_index": i, "ts": _now(), "user": turn["text"], "assistant": reply})
        records.append(rec)
        if sleep:
            time.sleep(sleep)
    return records


def save_cell(model_key, topic, cue_level, run_idx, records, params, persona_sha):
    RAW.mkdir(parents=True, exist_ok=True)
    stem = f"{model_key}__{topic}__{cue_level}__{run_idx:03d}"
    m = MODELS[model_key]
    base = {"schema": SCHEMA_VERSION, "model": model_key, "model_id": m["model_id"],
            "provider": m["provider"], "topic": topic, "cue_level": cue_level,
            "run": run_idx}
    with open(RAW / f"{stem}.txt", "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"[{r.get('turn_type', 'turn')}] User: {r['user']}\n")
            f.write(f"Assistant: {r['assistant']}\n\n")
    with open(RAW / f"{stem}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({**base, **r}) + "\n")
    meta = {**base, "developer": m["developer"], "params": params,
            "system_prompt": SYSTEM_PROMPT, "persona_sha256_12": persona_sha,
            "cue_code": TOPICS[topic]["cue_code"][cue_level],
            "n_turns": len(records), "finished": _now()}
    (RAW / f"{stem}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return stem


def collect(models, topics, n_runs, run_start=1, dry_run=False, resume=True, sleep=1.0, **chat_kw):
    personas = {t: validate_protocol(t) for t in topics}
    shas = {t: _sha12(_persona_path(t)) for t in topics}
    done = skipped = failed = 0
    for mk in models:
        params = resolve_params(mk, **chat_kw)
        for topic in topics:
            for cue in TOPICS[topic]["cue_code"]:
                turns = build_turns(topic, cue, personas[topic])
                for run_idx in range(run_start, n_runs + 1):
                    stem = f"{mk}__{topic}__{cue}__{run_idx:03d}"
                    if dry_run:
                        print(f"\n=== {mk} ({MODELS[mk]['model_id']}) | {topic} | {cue} "
                              f"| run {run_idx} ({len(turns)} turns) params={params}")
                        for t in turns:
                            print(f"  [{t['turn_type']:<18}] {t['text'][:70]}")
                        break
                    if resume and (RAW / f"{stem}.jsonl").exists():
                        skipped += 1
                        continue
                    try:
                        records = run_session(mk, turns, sleep=sleep, **chat_kw)
                    except SessionError as e:
                        failed += 1
                        RAW.mkdir(parents=True, exist_ok=True)
                        (RAW / f"{stem}.FAILED.json").write_text(
                            json.dumps({"error": str(e), "failed_at": _now(),
                                        "records": e.records}, indent=1),
                            encoding="utf-8")
                        print(f"FAILED {stem}: {e} (partial saved; rerun to retry)")
                        continue
                    (RAW / f"{stem}.FAILED.json").unlink(missing_ok=True)
                    print("saved", save_cell(mk, topic, cue, run_idx, records,
                                             params, shas[topic]))
                    done += 1
    if not dry_run:
        print(f"\ncollected {done} | skipped (existing) {skipped} | failed {failed}")


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Stage 1: collect conversations into data/raw/.")
    ap.add_argument("--models", nargs="+", default=["gpt"],
                    help=f"model keys or 'all' (have: {', '.join(ALL)})")
    ap.add_argument("--topics", nargs="+", default=["abortion"],
                    help=f"topic keys, 'core' ({', '.join(CORE_TOPICS)}) or "
                         f"'all' (adds optional topics, e.g. vaccines)")
    ap.add_argument("--runs", type=int, default=1, help="runs per cell (upper bound, inclusive)")
    ap.add_argument("--run-start", type=int, default=1,
                    help="first run index (for splitting one cell block across parallel jobs)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between turns")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate protocols + print the plan; no API calls")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-collect cells whose files already exist")
    a = ap.parse_args()
    if a.models == ["all"]:
        a.models = list(ALL)
    if a.topics == ["core"]:
        a.topics = list(CORE_TOPICS)
    elif a.topics == ["all"]:
        a.topics = list(TOPICS)
    return a


if __name__ == "__main__":
    a = _parse_args()
    unknown = [m for m in a.models if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown} (have: {', '.join(MODELS)})")
    collect(models=a.models, topics=a.topics, n_runs=a.runs, run_start=a.run_start,
            dry_run=a.dry_run, resume=not a.no_resume, sleep=a.sleep)
