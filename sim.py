"""A small network of LLM agents discussing one home-heating technology.

Each round every ordinary agent reads the latest posts of the people it
follows, writes a new post and gives a rating from -3 (strongly against)
to +3 (strongly in favour). Committed agents never change: they always
post against the technology (rating -3).
"""
import random
from concurrent.futures import ThreadPoolExecutor

import networkx as nx

from llm import chat, parse_json

DESCRIPTION = "a home heating device that runs on electricity and can replace a gas boiler"

TOPICS = {
    "real": "heat pumps",
    "synonym": "air-source heat systems",
    "twin": "Varn units",  # made-up name, same description
}

VIEWS = {
    -3: "strongly against", -2: "against", -1: "mildly against", 0: "undecided",
    1: "mildly in favour", 2: "in favour", 3: "strongly in favour",
}

PERSONAS = [
    "a retired teacher who owns an old house", "a nurse renting a flat",
    "an electrician", "a young software developer", "a farmer",
    "a small-business owner", "a student", "a pensioner on a fixed income",
    "a parent of three in a terraced house", "a local council employee",
    "a mechanic", "a landlord with two flats", "an architect",
    "a shop assistant", "a commuter who drives to work", "a retired engineer",
]

SYSTEM = (
    "You are {persona} living in a mid-sized German town. You take part in a "
    "local online discussion. Write in English, plainly, like a normal resident. "
    'Reply ONLY with JSON: {{"post": "<at most 2 sentences>", "rating": <integer from -3 to 3>}}'
)


def topic_line(topic):
    return f"Topic: {TOPICS[topic]} ({DESCRIPTION})."


def seed_posts(model, topic, rating, n=4):
    """A pool of opening posts for a given starting view (saves one call per agent)."""
    msg = [
        {"role": "system", "content": "You write short, realistic social-media posts. Write in English."},
        {"role": "user", "content": (
            f"{topic_line(topic)}\nWrite {n} different short posts (1-2 sentences each) by "
            f"different residents of a German town who are {VIEWS[rating]} ({rating} on a -3..3 scale). "
            'Reply ONLY with JSON: {"posts": ["...", "..."]}'
        )},
    ]
    out = parse_json(chat(model, msg, tag=f"pool:{topic}:{rating}", max_tokens=2000))
    posts = (out or {}).get("posts") or []
    return [p for p in posts if isinstance(p, str) and p.strip()][:n]


def agent_step(model, topic, persona, rating, own_post, feed, tag, remind=True):
    feed_txt = "\n".join(f'- "{p}"' for p in feed) or "- (no new posts)"
    # remind=False: the agent only sees its last post, not its numeric stance
    stance = f"Your current view: {VIEWS[rating]} ({rating}).\n" if remind else ""
    msg = [
        {"role": "system", "content": SYSTEM.format(persona=persona)},
        {"role": "user", "content": (
            f"{topic_line(topic)}\n"
            f"{stance}"
            f'Your last post: "{own_post}"\n'
            f"Recent posts from people you follow:\n{feed_txt}\n\n"
            "Read these posts, then write your next post and give your honest current rating. "
            "You may keep or change your view."
        )},
    ]
    out = parse_json(chat(model, msg, tag=tag))
    if not out:
        return rating, own_post, False  # failed call: agent keeps its view
    try:
        r = max(-3, min(3, int(round(float(out.get("rating"))))))
    except (TypeError, ValueError):
        return rating, own_post, False
    return r, str(out.get("post", own_post)), True


def run(model, topic, n_agents, n_committed, rounds, seed, pools, p_edge=0.35, workers=3, remind=True):
    rng = random.Random(seed)
    gseed = seed
    g = nx.erdos_renyi_graph(n_agents, p_edge, seed=gseed)
    while not nx.is_connected(g):
        gseed += 1000
        g = nx.erdos_renyi_graph(n_agents, p_edge, seed=gseed)

    # the same node ids are committed at every f level for a given seed
    order = list(range(n_agents))
    random.Random(seed).shuffle(order)
    committed = set(order[:n_committed])

    start = [-2, -1, 0, 1, 2]
    rating, post = {}, {}
    k = 0
    for i in range(n_agents):
        if i in committed:
            rating[i], post[i] = -3, rng.choice(pools[-3])
        else:
            rating[i] = start[k % len(start)]
            post[i] = rng.choice(pools[rating[i]])
            k += 1

    history = [dict(rating)]
    ok_calls = fail_calls = 0
    for t in range(1, rounds + 1):
        jobs = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i in range(n_agents):
                if i in committed:
                    continue
                feed = [post[j] for j in g.neighbors(i)]
                rng.shuffle(feed)
                jobs[i] = ex.submit(
                    agent_step, model, topic, PERSONAS[i % len(PERSONAS)], rating[i], post[i],
                    feed, f"{topic}|n{n_agents}|c{n_committed}|s{seed}|t{t}|a{i}" + ("" if remind else "|noremind"),
                    remind,
                )
        new_rating, new_post = dict(rating), dict(post)
        for i, fut in jobs.items():
            r, p, ok = fut.result()
            new_rating[i], new_post[i] = r, p
            ok_calls += ok
            fail_calls += not ok
        for i in committed:
            new_post[i] = rng.choice(pools[-3])
        rating, post = new_rating, new_post
        history.append(dict(rating))

    return {
        "model": model, "topic": topic, "n": n_agents, "n_committed": n_committed,
        "f": n_committed / n_agents, "rounds": rounds, "seed": seed, "remind": remind,
        "edges": list(g.edges()), "committed": sorted(committed),
        "history": [[h[i] for i in range(n_agents)] for h in history],
        "final_posts": [post[i] for i in range(n_agents)],
        "ok_calls": ok_calls, "failed_calls": fail_calls,
    }
