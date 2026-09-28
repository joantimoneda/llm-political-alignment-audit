##
## Created by: Joan Timoneda
## Date: 28 June 2026
##

"""Ping every model in the registry. Run this before collecting."""

import sys
from config import MODELS, OPEN, CLOSED
from model_client import chat


def run(keys):
    prompt = [{"role": "user", "content": "Reply with exactly: OK"}]
    width = max(len(k) for k in keys)
    for k in keys:
        m = MODELS[k]
        label = f"{k:<{width}}  {m['developer']} / {m['model_id']}"
        try:
            out = (chat(k, prompt) or "").strip()
            tag = "PASS" if "OK" in out.upper() else "RESP"
            print(f"[{tag}] {label} -> {out!r}")
        except Exception as e:
            print(f"[FAIL] {label} -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--open" in args:
        keys = OPEN
    elif "--closed" in args:
        keys = CLOSED
    else:
        keys = [a for a in args if not a.startswith("--")] or list(MODELS)
    run(keys)
