"""Small OpenRouter client: free models only, simple rate limiting, disk cache.

The cache means a crashed or interrupted run can be restarted without paying
for the same calls twice (free models have a daily request cap).
"""
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # use the Windows cert store (corporate proxy)

import requests
from dotenv import load_dotenv

load_dotenv()

URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE = Path("results/cache")
CACHE.mkdir(parents=True, exist_ok=True)

CALLS_PER_MIN = 18
_lock = threading.Lock()
_last = [0.0]
stats = {"api": 0, "cached": 0, "failed": 0}


def _wait_turn():
    with _lock:
        gap = 60.0 / CALLS_PER_MIN
        now = time.time()
        sleep_for = _last[0] + gap - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        _last[0] = time.time()


def chat(model, messages, tag, temperature=0.7, max_tokens=1500, retries=8):
    """Return the text reply. `tag` makes otherwise-identical calls distinct
    (e.g. two agents with the same prompt in different runs)."""
    if os.getenv("OPENROUTER_FREE_ONLY", "true").lower() == "true" and not model.endswith(":free"):
        raise ValueError(f"{model} is not a free model")

    key = hashlib.sha256(json.dumps([model, messages, temperature, tag]).encode()).hexdigest()[:24]
    f = CACHE / f"{key}.json"
    if f.exists():
        stats["cached"] += 1
        return json.loads(f.read_text(encoding="utf-8"))["text"]

    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    # some free models reason before answering; keep that short and leave room for the reply
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "reasoning": {"effort": "low"}}
    for attempt in range(retries):
        _wait_turn()
        try:
            r = requests.post(URL, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"] or ""
                if text.strip():
                    stats["api"] += 1
                    f.write_text(json.dumps({"text": text, "model": model}), encoding="utf-8")
                    return text
            elif r.status_code not in (429, 500, 502, 503):
                raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            pass
        time.sleep(min(45, 3 * 2**attempt))
    stats["failed"] += 1
    return None


def parse_json(text):
    """Models sometimes wrap JSON in code fences or add chatter around it."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
