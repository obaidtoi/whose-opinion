"""Does the text say what the rating says?

Every agent reply has a post and a self-reported rating. Here a second model
reads a sample of posts blind (no rating shown) and rates the stance of each
one. Agreement between the two tells us whether a language model can label
stance from text, and whether agents write one thing and rate another.

    uv run python validate_stance.py [--judge google/gemma-4-31b-it:free]
"""
import argparse
import glob
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import llm
from llm import chat, parse_json

p = argparse.ArgumentParser()
p.add_argument("--judge", default="google/gemma-4-31b-it:free")
p.add_argument("--fallback", default="baidu/ernie-5.5-flash-preview:free")
p.add_argument("--per-level", type=int, default=45)
p.add_argument("--batch", type=int, default=15)
args = p.parse_args()

OUT = Path("results/stance_check")
(OUT / "figures").mkdir(parents=True, exist_ok=True)

# 1) every (post, self-rating) pair the agents produced, from the call cache
pairs, seen = [], set()
for f in sorted(glob.glob("results/cache/*.json")):
    j = parse_json(json.loads(Path(f).read_text(encoding="utf-8"))["text"])
    if not j or "post" not in j or "rating" not in j:
        continue
    try:
        r = int(round(float(j["rating"])))
    except (TypeError, ValueError):
        continue
    post = str(j["post"]).strip()
    if post and post not in seen and -3 <= r <= 3:
        seen.add(post)
        pairs.append({"post": post, "self": r})

# 2) a sample balanced over rating levels, so agreement isn't just "everyone is +1"
rng = random.Random(0)
by_level = {r: [x for x in pairs if x["self"] == r] for r in range(-3, 4)}
sample = []
for r, xs in by_level.items():
    rng.shuffle(xs)
    sample += xs[: args.per_level]
rng.shuffle(sample)

PROMPT = (
    "Below are short posts by residents of a German town about a home heating technology "
    "(an electric device that can replace a gas boiler; it may be called by different names). "
    "For each post, rate the author's stance toward the technology from -3 (strongly against) "
    "to +3 (strongly in favour), using only the text. Mixed or hesitant posts belong near 0.\n\n"
    "{posts}\n\n"
    'Reply ONLY with JSON: {{"ratings": [{{"id": 1, "stance": <integer -3..3>}}, ...]}}'
)


def judge(model, batch, k):
    posts = "\n".join(f"{i + 1}. {x['post']}" for i, x in enumerate(batch))
    out = parse_json(chat(model, [{"role": "user", "content": PROMPT.format(posts=posts)}],
                          tag=f"stance-judge:{k}", temperature=0.0, max_tokens=2500, retries=4))
    got = {}
    for item in (out or {}).get("ratings", []):
        try:
            got[int(item["id"])] = max(-3, min(3, int(round(float(item["stance"])))))
        except (KeyError, TypeError, ValueError):
            pass
    return got


# 3) judge in batches; fall back to the agents' own model if the judge is unavailable
model = args.judge
for k in range(0, len(sample), args.batch):
    batch = sample[k: k + args.batch]
    got = judge(model, batch, k)
    if not got and model != args.fallback:
        print(f"{model} unavailable, falling back to {args.fallback}", flush=True)
        model = args.fallback
        got = judge(model, batch, k)
    for i, x in enumerate(batch):
        x["judge"] = got.get(i + 1)
        x["judge_model"] = model
    print(f"batch {k // args.batch + 1}: {len(got)}/{len(batch)} rated by {model}", flush=True)

rated = [x for x in sample if x.get("judge") is not None]
with (OUT / "pairs.jsonl").open("w", encoding="utf-8") as fh:
    for x in sample:
        fh.write(json.dumps(x, ensure_ascii=False) + "\n")

# 4) agreement
s = np.array([x["self"] for x in rated])
j = np.array([x["judge"] for x in rated])


def ranks(v):
    order = v.argsort(kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    for val in np.unique(v):  # average ties
        r[v == val] = r[v == val].mean()
    return r


def weighted_kappa(a, b, k=7):
    O = np.zeros((k, k))
    for x, y in zip(a + 3, b + 3):
        O[x, y] += 1
    W = np.array([[(i - jj) ** 2 for jj in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum(), O


rho = float(np.corrcoef(ranks(s), ranks(j))[0, 1]) if len(s) > 2 else None
kappa, conf = weighted_kappa(s, j)
gap = j - s
summary = {
    "judge_model": sorted({x["judge_model"] for x in rated}),
    "agent_model": "baidu/ernie-5.5-flash-preview:free",
    "n_pairs_available": len(pairs), "n_sampled": len(sample), "n_rated": len(rated),
    "spearman": rho, "weighted_kappa": float(kappa),
    "exact": float((gap == 0).mean()), "within_1": float((np.abs(gap) <= 1).mean()),
    "mean_gap": float(gap.mean()),
    "mean_gap_when_self_negative": float(gap[s < 0].mean()) if (s < 0).any() else None,
    "mean_gap_when_self_positive": float(gap[s > 0].mean()) if (s > 0).any() else None,
    "mean_abs_self": float(np.abs(s).mean()), "mean_abs_judge": float(np.abs(j).mean()),
    "confusion": conf.astype(int).tolist(),
    "figure": "results/stance_check/figures/confusion.png",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# 5) one figure: self-rating vs blind text rating
fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
im = ax.imshow(conf, cmap="Blues", origin="lower")
lv = list(range(-3, 4))
ax.set_xticks(range(7), lv)
ax.set_yticks(range(7), lv)
ax.set_xlabel("stance read from the post (blind)")
ax.set_ylabel("agent's own rating")
for a in range(7):
    for b in range(7):
        if conf[a, b]:
            ax.text(b, a, int(conf[a, b]), ha="center", va="center", fontsize=7,
                    color="white" if conf[a, b] > conf.max() / 2 else "#1F2933")
ax.plot([-0.5, 6.5], [-0.5, 6.5], color="#eb6834", lw=0.8, ls="--")
for sp in ax.spines.values():
    sp.set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "figures" / "confusion.png", facecolor="white")

print(json.dumps({k: v for k, v in summary.items() if k != "confusion"}, indent=2))
print("calls", llm.stats)
