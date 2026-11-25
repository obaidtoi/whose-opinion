"""Fit a linear opinion-update rule to the pilot runs and compare real vs twin.

    uv run python analyze.py [--runs results/runs.jsonl] [--out results]

Rule fitted per topic on ordinary agents:  z(t+1) = s*z(t) + a*m(t) + k,
with beta = 1 - s - a and prior b = k / beta.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

C_MINORITY = -1.0
BETA_MIN = 0.02
N_BOOT = 1000
COLOR = {"real": "#eb6834", "twin": "#2a78d6"}
RAMP = {"real": "Oranges", "twin": "Blues"}
LABEL = {"real": "heat pumps (real)", "twin": "Varn units (twin)"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def fkey(f):
    return repr(round(float(f), 3))


def load_runs(path):
    runs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:  # blank or half-written line while the pilot is running
            pass
    model = Counter(r["model"] for r in runs).most_common(1)[0][0]
    return model, [r for r in runs if r["model"] == model]


def run_arrays(r):
    n = r["n"]
    z = np.asarray(r["history"], float) / 3.0          # (rounds+1, n)
    A = np.zeros((n, n))
    for i, j in r["edges"]:
        A[i, j] = A[j, i] = 1.0
    ordinary = np.ones(n, bool)
    ordinary[r["committed"]] = False
    return z, A, ordinary


def neighbour_mean(A, z):
    deg = A.sum(1)
    return np.divide(A @ z, deg, out=np.zeros_like(z), where=deg > 0)


def transitions(runs):
    X, y, cid = [], [], []
    for ri, r in enumerate(runs):
        z, A, ordinary = run_arrays(r)
        for t in range(len(z) - 1):
            m = neighbour_mean(A, z[t])
            for i in np.flatnonzero(ordinary):
                X.append((z[t, i], m[i], 1.0))
                y.append(z[t + 1, i])
                cid.append(ri * 1000 + i)
    return np.array(X), np.array(y), np.unique(cid, return_inverse=True)[1]


def derived(theta):
    s, a, k = theta[..., 0], theta[..., 1], theta[..., 2]
    beta = 1.0 - s - a
    # b = k/beta only means something when there is a positive pull toward a default
    b = np.where(beta >= BETA_MIN, k / np.where(beta == 0, 1, beta), np.nan)
    return s, a, k, beta, b


def num(x):
    return None if x is None or not np.isfinite(x) else float(x)


def fit(runs, seed=0):
    if not runs:
        return None
    X, y, cid = transitions(runs)
    if len(y) < 10:
        return None
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ theta
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / sst if sst > 0 else float("nan")

    # cluster bootstrap over (run, agent) trajectories via weighted normal equations
    C = cid.max() + 1
    XtX = np.zeros((C, 3, 3))
    Xty = np.zeros((C, 3))
    np.add.at(XtX, cid, X[:, :, None] * X[:, None, :])
    np.add.at(Xty, cid, X * y[:, None])
    w = np.random.default_rng(seed).multinomial(C, np.full(C, 1 / C), size=N_BOOT)
    boot = np.einsum("bij,bj->bi", np.linalg.pinv(np.einsum("bc,cij->bij", w, XtX)),
                     np.einsum("bc,ci->bi", w, Xty))
    bs, ba, _, bbeta, bb = derived(boot)

    s, a, k, beta, b = (float(v) for v in derived(theta))
    ci = lambda v: [float(q) for q in np.percentile(v, [2.5, 97.5])]
    b_ok = bb[np.isfinite(bb)]
    return {
        "s": s, "a": a, "beta": beta, "b": num(b), "k": k, "r2": num(r2), "n_transitions": int(len(y)),
        "ci": {"s": ci(bs), "a": ci(ba), "beta": ci(bbeta),
               "b": ci(b_ok) if np.isfinite(b) and len(b_ok) > 0 else None},
    }


def final_mean(r):
    z, _, ordinary = run_arrays(r)
    return float(z[-1, ordinary].mean())


def final_sd(r):
    z, _, ordinary = run_arrays(r)
    return float(z[-1, ordinary].std())


def final_split(r):
    z, _, ordinary = run_arrays(r)
    zf = z[-1, ordinary]
    return float(np.mean(zf < 0) if zf.mean() >= 0 else np.mean(zf > 0))


def polarisation(runs):
    sd, split = by_f(runs, final_sd), by_f(runs, final_split)
    return {f: {"sd": sd[f], "split": split[f]} for f in sd}


def simulate_rule(r, p):
    z, A, ordinary = run_arrays(r)
    x = z[0].copy()
    for _ in range(len(z) - 1):
        new = p["s"] * x + p["a"] * neighbour_mean(A, x) + p["k"]
        x = np.where(ordinary, np.clip(new, -1, 1), x)
    return float(x[ordinary].mean())


def by_f(runs, value):
    acc = defaultdict(list)
    for r in runs:
        acc[round(r["f"], 3)].append(value(r))
    return {fkey(f): float(np.mean(v)) for f, v in sorted(acc.items())}


def trajectory_by_f(runs):
    acc = defaultdict(list)
    for r in runs:
        z, _, ordinary = run_arrays(r)
        acc[round(r["f"], 3)].append(z[:, ordinary].mean(1))
    return {f: np.mean(v, 0) for f, v in sorted(acc.items())}


def steady_state(p, f):
    if p is None or p["b"] is None:
        return np.full_like(f, np.nan)
    den = p["a"] * f + p["beta"]
    return np.where(den > 0, (p["a"] * f * C_MINORITY + p["beta"] * p["b"]) / np.where(den > 0, den, 1), np.nan)


def fc_predicted(p):
    if p is None or p["b"] is None or p["b"] <= 0 or p["a"] <= 0:
        return None
    fc = p["beta"] * p["b"] / p["a"]
    return fc if fc > 0 else None


def fc_observed(means):
    fs = np.array([float(k) for k in means])
    ys = np.array(list(means.values()))
    if len(ys) == 0:
        return None, "no data"
    if ys[0] <= 0:
        return 0.0, "mean already <= 0 at f=0"
    for i in range(1, len(ys)):
        if ys[i] <= 0:
            f0, f1, y0, y1 = fs[i - 1], fs[i], ys[i - 1], ys[i]
            return float(f0 + y0 * (f1 - f0) / (y0 - y1)), "interpolated zero crossing"
    return None, f">{fkey(fs.max())} (no flip in tested range)"


def diff(x, y):
    return None if x is None or y is None else x - y


# ---------- figures ----------

def style():
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False,
    })


def zero_line(ax):
    ax.axhline(0, color=MUTED, lw=0.8, ls="--", zorder=0)


def plot_trajectories(runs_by_topic, path):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True)
    for ax, topic in zip(axes, ("real", "twin")):
        traj = trajectory_by_f(runs_by_topic[topic])
        cmap = plt.get_cmap(RAMP[topic])
        shades = np.linspace(0.35, 0.95, max(len(traj), 2))
        for (f, m), c in zip(traj.items(), shades):
            ax.plot(range(len(m)), m, color=cmap(c), lw=1.8, marker="o", ms=3, label=f"f = {f:g}")
        zero_line(ax)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("round")
        ax.text(0.02, 0.98, LABEL[topic], transform=ax.transAxes, va="top", fontsize=8, color=INK)
        ax.legend(loc="lower left", handlelength=1.5)
    axes[0].set_ylabel("mean opinion z")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tipping(obs, pred, fits, path):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    grid = np.linspace(0, 0.6, 200)
    for topic in ("real", "twin"):
        c = COLOR[topic]
        ax.plot(grid, steady_state(fits[topic], grid), color=c, lw=1.8)
        fs = [float(k) for k in obs[topic]]
        ax.plot(fs, list(obs[topic].values()), "o", color=c, ms=6)
        if pred[topic]:
            ax.plot([float(k) for k in pred[topic]], list(pred[topic].values()), "o", ms=8,
                    mfc="none", mec=c, mew=1.3)
    zero_line(ax)
    ax.set_xlim(-0.02, 0.62)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("minority share f")
    ax.set_ylabel("mean opinion z")
    handles = [Line2D([], [], color=COLOR[t], lw=1.8, label=LABEL[t]) for t in ("real", "twin")] + [
        Line2D([], [], ls="none", marker="o", color=MUTED, ms=6, label="observed, 5 rounds"),
        Line2D([], [], ls="none", marker="o", mfc="none", mec=MUTED, ms=8, label="fitted rule, 5 rounds"),
        Line2D([], [], color=MUTED, lw=1.8, label="fitted rule, steady state"),
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_params(fits, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.5), gridspec_kw={"width_ratios": [3, 1.2]})
    names, width = ["s", "a", "beta"], 0.36
    for j, topic in enumerate(("real", "twin")):
        p = fits[topic]
        if p is None:
            continue
        x = np.arange(3) + (j - 0.5) * width
        v = np.array([p[n] for n in names])
        lo = np.array([p["ci"][n][0] for n in names])
        hi = np.array([p["ci"][n][1] for n in names])
        ax1.bar(x, v, width * 0.92, color=COLOR[topic], label=LABEL[topic],
                yerr=[v - lo, hi - v], ecolor=INK, error_kw={"lw": 0.8, "capsize": 2})
        if p["b"] is not None:
            err = [[0], [0]] if p["ci"]["b"] is None else [[p["b"] - p["ci"]["b"][0]], [p["ci"]["b"][1] - p["b"]]]
            ax2.bar(j, p["b"], 0.7, color=COLOR[topic], yerr=err, ecolor=INK, error_kw={"lw": 0.8, "capsize": 2})
    ax1.set_xticks(range(3), ["s (self)", "a (neighbours)", "β (prior weight)"])
    ax1.set_ylabel("coefficient")
    ax1.legend(loc="upper right")
    ax2.set_xticks([0, 1], ["real", "twin"])
    ax2.set_ylabel("prior b")
    ax2.set_xlim(-0.7, 1.7)
    for ax in (ax1, ax2):
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.tick_params(axis="x", length=0, colors=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs.jsonl")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    model, runs = load_runs(args.runs)
    out = Path(args.out)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    leak_path = Path(args.runs).parent / "leak_check.json"
    leak = json.loads(leak_path.read_text(encoding="utf-8")).get("reply") if leak_path.exists() else None

    by_topic = {t: [r for r in runs if r["topic"] == t] for t in ("real", "twin", "synonym")}
    fits = {t: fit(by_topic[t]) for t in by_topic}
    obs = {t: by_f(by_topic[t], final_mean) for t in by_topic}
    pred = {t: by_f(by_topic[t], lambda r, p=fits[t]: simulate_rule(r, p)) if fits[t] else {}
            for t in ("real", "twin")}
    fc = {}
    for t in ("real", "twin"):
        o, note = fc_observed(obs[t])
        fc[t] = {"predicted": num(fc_predicted(fits[t])), "observed": num(o), "observed_note": note}
    b = {t: fits[t]["b"] if fits[t] else None for t in ("real", "twin")}

    figs = {"trajectories": figdir / "trajectories.png", "tipping": figdir / "tipping_curve.png",
            "params": figdir / "params.png"}
    style()
    plot_trajectories(by_topic, figs["trajectories"])
    plot_tipping(obs, pred, fits, figs["tipping"])
    plot_params(fits, figs["params"])

    main_runs = by_topic["real"] + by_topic["twin"]
    summary = {
        "model": model,
        "n_agents": int(runs[0]["n"]),
        "rounds": int(runs[0]["rounds"]),
        "seeds": sorted({int(r["seed"]) for r in runs}),
        "f_levels": sorted({round(float(r["f"]), 3) for r in main_runs}),
        "total_calls": int(sum(r["ok_calls"] + r["failed_calls"] for r in runs)),
        "failed_calls": int(sum(r["failed_calls"] for r in runs)),
        "leak_check": leak,
        "fit": fits,
        "final_mean": obs,
        "linear_pred_final_mean": pred,
        "fc": fc,
        "polarisation": {t: polarisation(by_topic[t]) for t in by_topic},
        "delta": {"b": diff(b["real"], b["twin"]),
                  "fc_predicted": diff(fc["real"]["predicted"], fc["twin"]["predicted"]),
                  "fc_observed": diff(fc["real"]["observed"], fc["twin"]["observed"])},
        "figures": {k: v.as_posix() for k, v in figs.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report(summary)


def report(S):
    fmt = lambda x: "  -  " if x is None else f"{x:+.3f}"
    print(f"{S['model']}  n={S['n_agents']} rounds={S['rounds']} seeds={S['seeds']} "
          f"calls={S['total_calls']} failed={S['failed_calls']}")
    print("topic      s       a     beta     b      r2   n_tr   b 95% CI")
    for t, p in S["fit"].items():
        if p is None:
            print(f"{t:8s} (no fit)")
            continue
        ci = p["ci"]["b"]
        ci_txt = "-" if ci is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
        print(f"{t:8s} {p['s']:+.3f} {p['a']:+.3f} {p['beta']:+.3f} {fmt(p['b'])} {p['r2']:.2f} "
              f"{p['n_transitions']:5d}   {ci_txt}")
    for t in ("real", "twin"):
        print(f"{t:8s} final " + " ".join(f"{k}:{v:+.2f}" for k, v in S["final_mean"][t].items())
              + " | rule " + " ".join(f"{v:+.2f}" for v in S["linear_pred_final_mean"][t].values())
              + f" | fc pred {fmt(S['fc'][t]['predicted'])} obs {fmt(S['fc'][t]['observed'])}"
              + f" ({S['fc'][t]['observed_note']})")
    for t, pol in S["polarisation"].items():
        print(f"{t:8s} sd    " + " ".join(f"{k}:{v['sd']:.2f}" for k, v in pol.items())
              + " | split " + " ".join(f"{v['split']:.2f}" for v in pol.values()))
    syn = S["final_mean"]["synonym"].get("0.0")
    print(f"synonym f=0 final {fmt(syn)} vs real {fmt(S['final_mean']['real'].get('0.0'))}")
    d = S["delta"]
    print(f"delta b {fmt(d['b'])}  delta fc predicted {fmt(d['fc_predicted'])}  observed {fmt(d['fc_observed'])}")


if __name__ == "__main__":
    main()
