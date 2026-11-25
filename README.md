# Whose opinion is it?

A small pilot on model priors in LLM opinion dynamics. Obaid Tamboli, November 2026.

When a network of LLM agents is used to simulate how opinions spread, some of what we see comes from the agents reading each other and some may come from what the model already thinks about the topic. This pilot tries to separate the two for three outcomes: consensus, polarisation and tipping points. Sixteen agents discuss a home-heating technology for five rounds and rate it from −3 to +3. The same discussion runs under the real name ("heat pumps") and under a made-up name with the same description ("Varn units"), and a committed minority that always opposes the technology is added in steps. I fit a simple linear update rule, z(t+1) = s·z(t) + a·m(t) + β·b, to each topic. The real-minus-twin difference in the default view b, and in the minority share needed to flip the group, is then a rough measure of how much of the tipping comes from the model's prior rather than from the interaction.

The write-up is in [reports/pilot_report.pdf](reports/pilot_report.pdf).

## How to run

```
uv sync
cp .env.example .env          # then put your OPENROUTER_API_KEY in .env
uv run python run_pilot.py    # runs the grid, appends to results/runs.jsonl
uv run python analyze.py      # fits the model, writes results/summary.json and figures
uv run python make_report.py  # builds reports/pilot_report.pdf
```

Only free OpenRouter models (ids ending in `:free`) are used. `llm.py` refuses any other model unless `OPENROUTER_FREE_ONLY=false` is set. Free models have rate limits and a daily cap, so the full grid takes a while. All replies are cached in `results/cache/`, and finished runs are skipped, so an interrupted run can simply be restarted.

`run_pilot.py --model <id>:free` runs the same grid with another free model, and `--seeds 1 2 3` adds seeds.

The second experiment (agents not shown their own rating) is:

```
uv run python run_pilot.py --no-remind --no-synonym --committed 0 4 --out results/nostance/runs.jsonl
uv run python analyze.py --runs results/nostance/runs.jsonl --out results/nostance
```

## Files

- `sim.py`: the discussion itself: personas, prompts, the random graph, the committed minority and one run of the simulation.
- `llm.py`: a small OpenRouter client with rate limiting, retries and a disk cache.
- `run_pilot.py`: the leak check (does the model recognise "Varn units"?) and the grid of topics, minority sizes and seeds.
- `analyze.py`: fits the update rule per topic, with bootstrap intervals, and computes the flip points and figures.
- `make_report.py`: turns `results/summary.json` into the PDF. All result sentences are written from the numbers.
- `tests/make_fake_runs.py`: synthetic runs from a known rule, to check that `analyze.py` recovers it.
- `results/runs.jsonl`: one line per run, with the rating history of every agent.
- `results/leak_check.json`: the model's answer to the leak check.
- `results/summary.json`, `results/figures/`: output of `analyze.py`.
- `reports/pilot_report.pdf`: the pilot write-up.

## Results

One free model (`baidu/ernie-5.5-flash-preview:free`), one seed, 16 agents, 5 rounds, 880 agent calls, none failed.

1. **The prompt decided the dynamics.** When each agent was reminded of its own rating every round, agents kept about 95% of their view per round (s ≈ 0.95). There was no measurable pull toward any default, so the prior could not be measured at all. Without the reminder, s fell to about 0.5.
2. **Without the reminder, the model's default showed up.** Agents drifted toward a default view (β ≈ 0.3–0.5, as strong as the pull from neighbours), and opinions converged: the sd fell from 0.47 to 0.13–0.25. The default was more favourable for "heat pumps" (b = +0.35, 95% CI [0.27, 0.48]) than for the identical made-up "Varn units" (b = +0.14, [0.08, 0.23]).
3. **The minority did not tip it.** With a quarter of agents committed against heat pumps, the group still ended in favour. With the reminder, the same minority flipped it.
4. **The twin is not clean.** The model had never heard of "Varn units", but guessed from the description that it is a heat pump. The twin removes the name, not the technology.

These are directions, not sizes: more models, seeds and longer runs are the obvious next step. Details and limits are in the PDF.
