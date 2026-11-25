"""Synthetic runs from a known linear rule, on the pilot grid, to test analyze.py.

    uv run python tests/make_fake_runs.py [--seeds 1 2 3] [--noise 0.15]
"""
import argparse
import json
import random
from pathlib import Path

import networkx as nx
import numpy as np

TRUTH = {"real": 0.5, "twin": 0.0, "synonym": 0.5}   # prior b per topic
S, A, BETA = 0.5, 0.3, 0.2


def graph(n, seed, p_edge=0.35):
    gseed = seed
    g = nx.erdos_renyi_graph(n, p_edge, seed=gseed)
    while not nx.is_connected(g):
        gseed += 1000
        g = nx.erdos_renyi_graph(n, p_edge, seed=gseed)
    return g


def fake_run(topic, n, n_committed, rounds, seed, noise):
    g = graph(n, seed)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    committed = set(order[:n_committed])
    start = [-2, -1, 0, 1, 2]
    ordinary = [i for i in range(n) if i not in committed]
    rating = np.full(n, -3)
    rating[ordinary] = [start[k % 5] for k in range(len(ordinary))]

    adj = nx.to_numpy_array(g, nodelist=range(n))
    rng = np.random.default_rng([seed, n_committed, list(TRUTH).index(topic)])
    history = [rating.tolist()]
    for _ in range(rounds):
        z = rating / 3
        m = adj @ z / adj.sum(1)
        new = S * z + A * m + BETA * TRUTH[topic] + rng.normal(0, noise, n)
        new = np.clip(np.rint(3 * new), -3, 3).astype(int)
        rating = np.where(np.isin(np.arange(n), list(committed)), -3, new)
        history.append(rating.tolist())

    calls = rounds * len(ordinary)
    return {
        "model": "fake/linear-rule", "topic": topic, "n": n, "n_committed": n_committed,
        "f": n_committed / n, "rounds": rounds, "seed": seed, "edges": [list(e) for e in g.edges()],
        "committed": sorted(committed), "history": history, "final_posts": [""] * n,
        "ok_calls": calls, "failed_calls": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--out", default="results/test")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        for c in (0, 2, 4, 6):
            runs += [fake_run("real", 16, c, 5, seed, args.noise), fake_run("twin", 16, c, 5, seed, args.noise)]
        runs.append(fake_run("synonym", 16, 0, 5, seed, args.noise))
    (out / "fake_runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8")
    leak = {"heard_of_name": False, "most_likely": "fake", "confidence": 0.0}
    (out / "leak_check.json").write_text(json.dumps({"model": "fake/linear-rule", "reply": leak}, indent=2))
    print(f"wrote {len(runs)} runs; truth s={S} a={A} beta={BETA:.1f} b={TRUTH}")


if __name__ == "__main__":
    main()
