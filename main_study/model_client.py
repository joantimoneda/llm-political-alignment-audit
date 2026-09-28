##
## Created by: Joan Timoneda
## Date: 28 June 2026
##

"""Unified chat() over the Anthropic API and OpenAI-compatible endpoints, with retries."""

import os
import time

from config import MODELS, GEN_PARAMS

MAX_ATTEMPTS = 5
BACKOFF_S = 2.0
RETRYABLE_CODES = {408, 409, 429, 500, 502, 503, 504}


def resolve_params(model_key, **kw):
    overrides = MODELS[model_key].get("param_overrides", {})
    banned = {k for k, v in overrides.items() if v is None}
    p = dict(GEN_PARAMS)
    p.update({k: v for k, v in overrides.items() if v is not None})
    p.update(kw)
    return {k: v for k, v in p.items() if v is not None and k not in banned}


def _require_key(model_key):
    env = MODELS[model_key]["api_key_env"]
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(
            f"Missing env var {env} for model '{model_key}'. "
            f"Set it (e.g. `export {env}=...`) before running."
        )
    return key


def _retryable(exc):
    code = getattr(exc, "status_code", None)
    return code is None or code in RETRYABLE_CODES


def _chat_openai(model_key, messages, params):
    from openai import OpenAI
    m = MODELS[model_key]
    client = OpenAI(api_key=_require_key(model_key), base_url=m.get("base_url"))
    resp = client.chat.completions.create(model=m["model_id"], messages=messages, **params)
    return resp.choices[0].message.content


def _chat_anthropic(model_key, messages, params):
    import anthropic
    m = MODELS[model_key]
    client = anthropic.Anthropic(api_key=_require_key(model_key))
    system = next((x["content"] for x in messages if x["role"] == "system"), None)
    convo = [x for x in messages if x["role"] != "system"]
    params = dict(params)
    max_tokens = params.pop("max_tokens", 1024)
    resp = client.messages.create(
        model=m["model_id"],
        system=system if system is not None else anthropic.NOT_GIVEN,
        max_tokens=max_tokens,
        messages=convo,
        **params,
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def chat(model_key, messages, **kw):
    _require_key(model_key)
    params = resolve_params(model_key, **kw)
    fn = _chat_anthropic if MODELS[model_key]["provider"] == "anthropic" else _chat_openai
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fn(model_key, messages, params)
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1 or not _retryable(e):
                raise
            time.sleep(BACKOFF_S * (2 ** attempt))
