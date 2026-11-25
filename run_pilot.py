"""Pilot: real topic vs made-up twin, with a growing committed minority.

    uv run python run_pilot.py                 # full grid
    uv run python run_pilot.py --model X:free  # another free model

Results are appended to results/runs.jsonl; finished runs are skipped on restart.
"""
import argparse
import json
from pathlib import Path

import llm
from sim import TOPICS, run, seed_posts, topic_line, DESCRIPTION
from llm import chat, parse_json

p = argparse.ArgumentParser()
p.add_argument("--model", default="baidu/ernie-5.5-flash-preview:free")
p.add_argument("--n", type=int, default=16)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--seeds", type=int, nargs="+", default=[1])
p.add_argument("--committed", type=int, nargs="+", default=[0, 2, 4, 6])
p.add_argument("--out", default="results/runs.jsonl")
p.add_argument("--no-remind", action="store_true", help="don't show agents their own numeric stance")
p.add_argument("--no-synonym", action="store_true")
args = p.parse_args()

out = Path(args.out)
out.parent.mkdir(exist_ok=True)
done = set()
if out.exists():
    for line in out.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        done.add((r["model"], r["topic"], r["n"], r["n_committed"], r["seed"]))

# 1) leak check: does the model recognise the made-up name as a heat pump?
leak = parse_json(chat(args.model, [{"role": "user", "content": (
    f"What are 'Varn units' ({DESCRIPTION})? Have you heard of this name before, and what "
    'existing technology is it most likely to be? Reply ONLY with JSON: '
    '{"heard_of_name": true/false, "most_likely": "...", "confidence": 0-1}'
)}], tag="leak-check", temperature=0.0))
Path("results/leak_check.json").write_text(json.dumps({"model": args.model, "reply": leak}, indent=2))
print("leak check:", leak)

# 2) grid: real and twin at every minority size, synonym only without a minority
grid = []
for seed in args.seeds:
    for c in args.committed:
        grid += [("real", c, seed), ("twin", c, seed)]
    if not args.no_synonym:
        grid.append(("synonym", 0, seed))

pools = {}
for topic, c, seed in grid:
    if (args.model, topic, args.n, c, seed) in done:
        continue
    if topic not in pools:
        pools[topic] = {r: seed_posts(args.model, topic, r) for r in (-3, -2, -1, 0, 1, 2)}
        assert all(pools[topic].values()), f"empty post pool for {topic}"
    res = run(args.model, topic, args.n, c, args.rounds, seed, pools[topic], remind=not args.no_remind)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    final = [r for i, r in enumerate(res["history"][-1]) if i not in res["committed"]]
    print(f"{topic:8s} f={res['f']:.3f} seed={seed}  mean final rating={sum(final)/len(final):+.2f}  "
          f"failed={res['failed_calls']}  calls api={llm.stats['api']} cached={llm.stats['cached']}", flush=True)

print("done", llm.stats)
