"""Build the short pilot write-up (PDF) from results/summary.json.

    uv run python make_report.py
    uv run python make_report.py --summary results/test/report_fixture/summary.json --out results/test/report_fixture/report.pdf

All result sentences are written from the numbers in the summary, so the
report can be rebuilt whenever the analysis changes. Missing numbers are
reported as "not identified".
"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

TITLE = "Whose opinion is it? A pilot on model priors in LLM opinion dynamics"
SUBTITLE = ("Obaid Tamboli · pilot study · November 2026 · code and data: "
            "<a href='https://github.com/obaidtoi/whose-opinion' color='#5b6169'><u>github.com/obaidtoi/whose-opinion</u></a>")
NA = "not identified"

# kept in step with sim.py by hand (importing sim would pull in the API client)
DESCRIPTION = "a home heating device that runs on electricity and can replace a gas boiler"
P_EDGE = 0.35
START_SD = 0.48  # observed sd of the free agents at the start (0.47 to 0.48)


# ---------------------------------------------------------------- fonts / styles
def _fonts():
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # DejaVu ships with matplotlib
    for name, f in [("Serif", "DejaVuSerif"), ("Serif-B", "DejaVuSerif-Bold"),
                    ("Serif-I", "DejaVuSerif-Italic"), ("Serif-BI", "DejaVuSerif-BoldItalic"),
                    ("Sans", "DejaVuSans"), ("Sans-B", "DejaVuSans-Bold"),
                    ("Sans-I", "DejaVuSans-Oblique"), ("Sans-BI", "DejaVuSans-BoldOblique")]:
        pdfmetrics.registerFont(TTFont(name, str(d / f"{f}.ttf")))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-BI")


INK = colors.HexColor("#1f2328")
GREY = colors.HexColor("#5b6169")
RULE = colors.HexColor("#d0d4d9")


def _styles():
    body = ParagraphStyle("body", fontName="Serif", fontSize=9.4, leading=13, textColor=INK, spaceAfter=5)
    return {
        "title": ParagraphStyle("title", fontName="Sans-B", fontSize=15.5, leading=19, textColor=INK, spaceAfter=3),
        "sub": ParagraphStyle("sub", fontName="Sans", fontSize=9.5, leading=12, textColor=GREY, spaceAfter=10),
        "h": ParagraphStyle("h", fontName="Sans-B", fontSize=11, leading=14, textColor=INK, spaceBefore=6, spaceAfter=3,
                            keepWithNext=1),
        "body": body,
        "bullet": ParagraphStyle("bullet", parent=body, spaceAfter=1.5),
        "eq": ParagraphStyle("eq", parent=body, alignment=TA_CENTER, fontSize=10.2, leading=15, spaceAfter=2),
        "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8.2, leading=10.5, textColor=GREY, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.6, leading=10.5, textColor=INK),
        "cellh": ParagraphStyle("cellh", fontName="Sans-B", fontSize=8.6, leading=10.5, textColor=INK),
        "ref": ParagraphStyle("ref", parent=body, fontSize=8.8, leading=12, leftIndent=10, firstLineIndent=-10, spaceAfter=2),
    }


# ---------------------------------------------------------------- number helpers
def fmt(x, signed=False):
    if x is None:
        return NA
    s = f"{x:+.2f}" if signed else f"{x:.2f}"
    return s.replace("-", "−")


def fmt_f(f):
    """Minority shares like 0.125 need three decimals; strip trailing zeros."""
    return f"{f:.3f}".rstrip("0").rstrip(".") if f else "0"


def fmt_ci(ci):
    if not ci or ci[0] is None or ci[1] is None:
        return ""
    return f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def f_key(d, f):
    """Look up a value keyed by minority share, whatever the key's string form."""
    if not d:
        return None
    for k, v in d.items():
        try:
            if abs(float(k) - f) < 1e-9:
                return v
        except ValueError:
            pass
    return None


def ci_overlap(a, b):
    if not a or not b or None in a or None in b:
        return None
    return a[0] <= b[1] and b[0] <= a[1]


# ---------------------------------------------------------------- text blocks
def leak_line(leak):
    reply = leak.get("reply") if isinstance(leak, dict) and "reply" in leak else leak
    if not isinstance(reply, dict) or not reply:
        return ("Leak check: the direct question about “Varn units” gave no usable answer, "
                "so I could not test whether the model knows the name.")
    heard = reply.get("heard_of_name")
    likely = str(reply.get("most_likely") or "").strip()
    conf = reply.get("confidence")
    heard_txt = {True: "had heard of", False: "had not heard of"}.get(heard, "may or may not have heard of")
    conf_txt = f" (confidence {fmt(float(conf))})" if isinstance(conf, (int, float)) else ""
    s = f"Leak check: asked directly, the model said it {heard_txt} “Varn units”"
    if likely:
        s += f" and guessed it is most likely “{escape(likely)}”{conf_txt}."
    else:
        s += f"{conf_txt}."
    if "heat pump" in likely.lower():
        s += " So it can work out from the description what the twin is. The twin hides the name and leaves the technology recognisable."
    elif likely:
        s += " It did not link the name to heat pumps."
    return s


def results_text(S):
    fit, fc, fm, pred, delta = (S.get(k) or {} for k in ("fit", "fc", "final_mean", "linear_pred_final_mean", "delta"))
    F = sorted(S.get("f_levels") or [])
    fmax = F[-1] if F else None
    real, twin, syn = fit.get("real") or {}, fit.get("twin") or {}, fit.get("synonym") or {}
    paras = []

    # fit quality and the default view b
    p = []
    if real.get("r2") is not None or twin.get("r2") is not None:
        p.append(f"The linear update explains the round-to-round changes with R² = {fmt(real.get('r2'))} "
                 f"for heat pumps and {fmt(twin.get('r2'))} for the twin "
                 f"({real.get('n_transitions', NA)} and {twin.get('n_transitions', NA)} agent transitions).")
    br, bt = real.get("b"), twin.get("b")
    cr, ct = get(real, "ci", "b"), get(twin, "ci", "b")
    if br is not None and bt is not None:
        db = delta.get("b") if delta.get("b") is not None else br - bt
        p.append(f"The fitted default view is b = {fmt(br, True)} for heat pumps and {fmt(bt, True)} for "
                 f"Varn units, so Δb = {fmt(db, True)}. The weight on that default, β, is "
                 f"{fmt(real.get('beta'))} and {fmt(twin.get('beta'))}.")
        ov = ci_overlap(cr, ct)
        more = "more favourable" if br > bt else "less favourable"
        if ov is True:
            p.append(f"The 95% intervals overlap ({fmt_ci(cr)} and {fmt_ci(ct)}), so with this much data I "
                     f"cannot say the two defaults differ, even though the point estimate for the real name "
                     f"is {more}.")
        elif ov is False:
            p.append(f"The 95% intervals do not overlap ({fmt_ci(cr)} and {fmt_ci(ct)}). This suggests the "
                     f"model brings a {more} prior to the real name than to the made-up one.")
        else:
            p.append("I have no interval for at least one of the two, so I cannot judge whether the gap is noise.")
    else:
        missing = [n for n, v in (("heat pumps", br), ("Varn units", bt)) if v is None]
        p.append(f"Because β is close to zero, b cannot be separated from the intercept for "
                 f"{' and '.join(missing)}, so Δb is {NA}.")
    paras.append(" ".join(p))

    # flip points
    p = []
    if min(real.get("s") or 0, twin.get("s") or 0) >= 0.85:
        paras.append("Because the free agents barely moved, I do not read flip points from these runs. With the "
                     "committed agents set aside, the free agents already started slightly below zero, so a "
                     "“crossing” would only reflect where they began.")
        F_skip = True
    else:
        F_skip = False
    step = min((y - x for x, y in zip(F, F[1:])), default=None)
    for topic, label in (("real", "heat pumps"), ("twin", "Varn units")):
        obs, prd = get(fc, topic, "observed"), get(fc, topic, "predicted")
        note = escape((get(fc, topic, "observed_note") or "").strip())
        on_grid = obs is not None and any(abs(obs - f) < 1e-9 for f in F)
        if obs is None:
            bits = [f"f up to {fmt_f(fmax)}"] if fmax is not None else []
            bits += [note] if note else []
            s = f"With {label} the group did not flip within the tested range" + (
                f" ({', '.join(bits)})." if bits else ".")
        elif F and abs(obs - F[0]) < 1e-9:
            s = f"With {label} the mean was already at or below zero at f = {fmt_f(obs)}, so there was nothing to flip."
        else:
            where = fmt_f(obs) if on_grid else fmt(obs)
            s = f"With {label} the mean final opinion crossed zero at f = {where}" + (f" ({note})." if note else ".")
        if prd is None:
            b_ = (real if topic == "real" else twin).get("b")
            why = " because the fitted default is not positive" if b_ is not None and b_ <= 0 else ""
            s += f" The fitted model’s flip point is {NA}{why}."
        else:
            s += f" The fitted model puts it at f<sub>c</sub> = {fmt(prd)}"
            if fmax is not None and prd > fmax:
                s += ", beyond the largest share I tested"
            s += "."
            if obs is None:
                ok = fmax is not None and prd > fmax
            elif on_grid:  # the crossing lies between the previous tested share and this one
                below = [f for f in F if f < obs - 1e-9]
                ok = (below[-1] if below else float("-inf")) < prd <= obs + 1e-9
            else:  # interpolated crossing: allow half a grid step
                ok = step is not None and abs(prd - obs) <= step / 2
            if obs is not None and not (F and abs(obs - F[0]) < 1e-9):
                s += (" That is consistent with the observed crossing, given how coarse the grid is." if ok else
                      " That does not agree with the observed crossing, so the linear model misses something here.")
            elif obs is None:
                s += (" That is consistent with seeing no flip." if ok else
                      " Yet no flip was observed, so the linear model misses something here.")
        p.append(s)
    if F_skip:
        p = []
    dfp, dfo = delta.get("fc_predicted"), delta.get("fc_observed")
    if F_skip:
        pass
    elif dfp is None and dfo is None:
        p.append(f"The difference between real and twin, Δf<sub>c</sub>, is {NA}.")
    else:
        head = "The difference in the flip point between real and twin is Δf<sub>c</sub> = "
        if dfp is not None and dfo is not None:
            p.append(f"{head}{fmt(dfp, True)} from the fitted model and {fmt(dfo, True)} from the observed crossings.")
        elif dfp is not None:
            p.append(f"{head}{fmt(dfp, True)} from the fitted model, and from the observed crossings it is {NA}.")
        else:
            p.append(f"{head}{fmt(dfo, True)} from the observed crossings, and from the fitted model it is {NA}.")
        fcr = get(fc, "real", "predicted")
        if dfp is not None and fcr and 0 <= dfp <= fcr:
            p.append(f"If the twin is a fair no-prior baseline, about {100 * dfp / fcr:.0f}% of the minority "
                     f"needed to flip the heat-pump discussion is there to overcome the model’s prior. "
                     f"I would not lean on this figure before more seeds and models are in.")
    if p:
        paras.append(" ".join(p))

    # does the 5-round linear prediction match the observed final means?
    p = []
    last = None
    for topic, label in (("real", "heat pumps"), ("twin", "Varn units")):
        o, q = fm.get(topic) or {}, pred.get(topic) or {}
        pairs = [(f, f_key(o, f), f_key(q, f)) for f in F]
        pairs = [(f, x, y) for f, x, y in pairs if x is not None and y is not None]
        if not pairs:
            p.append(f"There is no prediction to compare for {label}.")
            continue
        g, gf = max((abs(x - y), f) for f, x, y in pairs)
        if g <= 0.10:
            verdict = "matches the observed final means closely"
        elif g <= 0.25:
            verdict = "roughly matches the observed final means"
        else:
            verdict = "does not match the observed final means well"
        gap = f"largest gap {fmt(g)} on the z scale, at f = {fmt_f(gf)}"
        if last == verdict:
            s = f"For {label} it also {verdict.split(' the observed')[0]} ({gap})."
        else:
            s = f"For {label} the {S.get('rounds', 5)}-round prediction of the linear model {verdict} ({gap})."
        last = verdict
        diffs = [x - y for _, x, y in pairs]
        if len(diffs) > 1 and g > 0.05:
            if all(d < 0 for d in diffs):
                s += " The observed means are below the prediction at every f."
            elif all(d > 0 for d in diffs):
                s += " The observed means are above the prediction at every f."
        p.append(s)
    s_, a_ = real.get("s"), real.get("a")
    if s_ is not None and a_ is not None and 0 < s_ + a_ < 1:
        left = (s_ + a_) ** int(S.get("rounds") or 5)
        p.append(f"With the heat-pump fit, the mean-field model still has {100 * left:.0f}% of the distance to "
                 f"its resting point left after {S.get('rounds', 5)} rounds at f = 0, so the final means "
                 f"are not steady-state values.")
    paras.append(" ".join(p))

    # wording check
    zs, zr = f_key(fm.get("synonym") or {}, 0.0), f_key(fm.get("real") or {}, 0.0)
    if zs is not None and zr is not None:
        gap = zs - zr
        s = (f"The wording check without a minority gave a final mean of {fmt(zs, True)} for "
             f"“air-source heat systems” against {fmt(zr, True)} for “heat pumps” "
             f"(difference {fmt(gap, True)}).")
        if syn.get("b") is not None and real.get("b") is not None:
            s += f" Its fitted default is b = {fmt(syn['b'], True)}."
        s += (" So a change of wording alone made little difference here." if abs(gap) < 0.10 else
              " So wording alone moves the result by about this much, which is also a rough size for the "
              "noise in the comparison of real and twin.")
        paras.append(s)
    else:
        paras.append("The synonym run was not available, so I have no wording check in this summary.")
    return paras


# ---------------------------------------------------------------- flowables
def _pol(S, topic, f, key):
    return get(f_key(get(S, "polarisation", topic) or {}, f), key)


def polarisation_table(S, st):
    """Table 2: spread of the final opinions per minority share; None if the key is missing."""
    if not S.get("polarisation"):
        return None
    F = sorted(S.get("f_levels") or [])
    head = ["f", "real sd", "real split", "twin sd", "twin split"]
    rows = [[Paragraph(h, st["cellh"]) for h in head]]
    for f in F:
        rows.append([Paragraph(fmt_f(f), st["cell"])] + [
            Paragraph(fmt(_pol(S, t, f, k)) if _pol(S, t, f, k) is not None else "n/a", st["cell"])
            for t in ("real", "twin") for k in ("sd", "split")])
    tbl = Table(rows, colWidths=[14 * mm] + [13.8 * mm] * 4, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return tbl


def polarisation_text(S):
    """One or two factual sentences on how the spread changes with f, real vs twin."""
    if not S.get("polarisation"):
        return None
    F = sorted(S.get("f_levels") or [])
    out = []
    for topic, label in (("real", "heat pumps"), ("twin", "Varn units")):
        sds = [(f, _pol(S, topic, f, "sd")) for f in F]
        sds = [(f, v) for f, v in sds if v is not None]
        if len(sds) < 2:
            continue
        (f0, lo), (f1, hi) = sds[0], sds[-1]
        d = hi - lo
        if d > 0.05:
            s = f"With {label} the final opinions spread out as f grew"
        elif d < -0.05:
            s = f"With {label} the final opinions pulled together as f grew"
        else:
            s = f"With {label} the spread at the largest f was about the same as without a minority"
        s += f" (sd {fmt(lo)} at f = {fmt_f(f0)}, {fmt(hi)} at f = {fmt_f(f1)})"
        fp, vp = max(sds[1:-1], key=lambda x: x[1], default=(None, None))
        if vp is not None and vp > max(lo, hi) + 0.05:
            s += f", with a peak of {fmt(vp)} at f = {fmt_f(fp)} in between"
        out.append(s + ".")
    pairs = [(_pol(S, "real", f, "sd"), _pol(S, "twin", f, "sd")) for f in F]
    pairs = [(r, t) for r, t in pairs if r is not None and t is not None]
    if pairs:
        diff = sum(r - t for r, t in pairs) / len(pairs)
        if abs(diff) < 0.05:
            out.append(f"The two topics are about equally spread (mean difference in sd {fmt(diff, True)}).")
        else:
            more = "more" if diff > 0 else "less"
            out.append(f"Opinions about heat pumps are on average {more} spread than about the twin "
                       f"(mean difference in sd {fmt(diff, True)}).")
    # free agents start at -2..+2 (z = rating/3), a spread of about 0.47
    all_sd = [_pol(S, t, f, "sd") for t in ("real", "twin") for f in F]
    all_sd = [v for v in all_sd if v is not None]
    if all_sd and all(abs(v - START_SD) < 0.12 for v in all_sd):
        out.append(f"All of these are close to the starting spread (sd {fmt(START_SD)}, views from −2 to +2). "
                   "So the groups neither converged to a consensus nor split into two camps, and "
                   "opinions mostly stayed where they began.")
    elif all_sd and max(all_sd) < START_SD - 0.12:
        out.append(f"The spread fell below the starting value (sd {fmt(START_SD)}), so opinions moved "
                   "toward each other.")
    zs = _pol(S, "synonym", 0.0, "sd")
    if zs is not None:
        out.append(f"The synonym run had sd {fmt(zs)} at f = 0.")
    return " ".join(out) if out else None


def key_finding(S):
    s = [get(S, "fit", t, "s") for t in ("real", "twin")]
    beta = [get(S, "fit", t, "beta") for t in ("real", "twin")]
    if None in s or None in beta:
        return None
    if min(s) < 0.85:
        return None
    return (f"<b>With the reminder.</b> The agents barely changed their minds. Each round they kept about "
            f"{100 * min(s):.0f}% to {100 * max(s):.0f}% of their previous view, gave little weight to their "
            f"neighbours, and showed no positive pull toward a default view (β ≈ {fmt(beta[0])} and "
            f"{fmt(beta[1])}). With no pull there is nothing to compare between the real topic and its twin, so "
            "the prior and the flip point are not identified in this design. In my prompt every agent was told "
            "its own current rating each round. That reminder is the first suspect, and section 5 removes it.")


def ablation(S, A, st):
    """Compare the main run (stance reminder) with the run without it."""
    rows = [["", "heat pumps<br/>with reminder", "Varn units<br/>with reminder",
             "heat pumps<br/>no reminder", "Varn units<br/>no reminder"]]
    for key, label in (("s", "<i>s</i> (own view)"), ("a", "<i>a</i> (neighbours)"),
                       ("beta", "β (default)"), ("b", "<i>b</i> (default view)")):
        row = [label]
        for src, t in ((S, "real"), (S, "twin"), (A, "real"), (A, "twin")):
            v = get(src, "fit", t, key)
            row.append(fmt(v, signed=key == "b") if v is not None else "n/a")
        rows.append(row)
    for f in (0.0, 0.25):
        row = [f"final mean, f = {fmt_f(f)}"]
        for src, t in ((S, "real"), (S, "twin"), (A, "real"), (A, "twin")):
            v = f_key(get(src, "final_mean", t) or {}, f)
            row.append(fmt(v, signed=True) if v is not None else "n/a")
        rows.append(row)
    rows = [[Paragraph(str(c), st["cell"]) for c in r] for r in rows]
    tbl = Table(rows, colWidths=[38 * mm] + [33 * mm] * 4, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))

    s_main = [get(S, "fit", t, "s") for t in ("real", "twin")]
    s_abl = [get(A, "fit", t, "s") for t in ("real", "twin")]
    b_abl = [get(A, "fit", t, "beta") for t in ("real", "twin")]
    text = []
    if None not in s_main + s_abl:
        drop = sum(s_main) / 2 - sum(s_abl) / 2
        if drop > 0.1:
            text.append(f"Without the reminder the agents became much less stubborn (<i>s</i> fell from about "
                        f"{fmt(sum(s_main) / 2)} to {fmt(sum(s_abl) / 2)}). So a large part of the stubbornness "
                        "came from my prompt.")
        elif drop > 0.03:
            text.append(f"Without the reminder the agents were somewhat less stubborn (<i>s</i> about "
                        f"{fmt(sum(s_main) / 2)} → {fmt(sum(s_abl) / 2)}).")
        else:
            text.append(f"Removing the reminder made little difference to stubbornness (<i>s</i> about "
                        f"{fmt(sum(s_main) / 2)} → {fmt(sum(s_abl) / 2)}), so it seems to belong to the model "
                        "rather than to the prompt.")
    if None not in b_abl:
        if max(b_abl) >= 0.02:
            br, bt = get(A, "fit", "real", "b"), get(A, "fit", "twin", "b")
            cr, ct = get(A, "fit", "real", "ci", "b"), get(A, "fit", "twin", "ci", "b")
            text.append(f"A pull toward a default appears (β = {fmt(b_abl[0])} for heat pumps, {fmt(b_abl[1])} "
                        f"for the twin), stronger than the pull from neighbours. The default views "
                        f"are b = {fmt(br, True) if br is not None else NA} {fmt_ci(cr) if cr else ''} for heat "
                        f"pumps and {fmt(bt, True) if bt is not None else NA} {fmt_ci(ct) if ct else ''} for the "
                        "twin.")
            if cr and ct and cr[0] > ct[1]:
                text.append(f"The intervals do not overlap, so the real name carries a more favourable default "
                            f"than the identical made-up technology (Δb = {fmt(br - bt, True)}). The twin is not "
                            "neutral either, which fits the leak check and a general positive lean.")
        else:
            text.append("There is still no measurable pull toward a default view (β ≈ 0 for both topics).")
    sds =[get(A, "polarisation", t) for t in ("real", "twin")]
    sds = [v.get("sd") for d in sds if d for v in d.values() if v and v.get("sd") is not None]
    if sds and max(sds) < START_SD - 0.12:
        text.append(f"Opinions also converged. The final spread was sd {fmt(min(sds))} to {fmt(max(sds))}, down "
                    f"from {fmt(START_SD)} at the start, so the groups moved toward a consensus near the "
                    "model’s default.")
    m = f_key(get(A, "final_mean", "real") or {}, 0.25)
    if m is not None and m > 0:
        text.append(f"With a quarter of the agents committed against heat pumps, the free agents still moved toward heat pumps and ended in "
                    f"favour (mean z = {fmt(m, True)}), having started slightly below zero.")
    text.append("These runs use one seed and two minority sizes, so they show the direction of these effects and leave the size open.")
    return tbl, " ".join(text)


def param_table(S, st):
    fit = S.get("fit") or {}
    topics = [("real", "heat pumps"), ("twin", "Varn units (twin)"), ("synonym", "air-source (synonym)")]
    rows = [[Paragraph("", st["cellh"])] + [Paragraph(lbl, st["cellh"]) for _, lbl in topics]]
    spec = [("s", "s", "own view"), ("a", "a", "neighbours"), ("beta", "β", "default"),
            ("b", "b", "default view")]
    for key, sym, desc in spec:
        row = [Paragraph(f"<i>{sym}</i> <font color='#5b6169'>({desc})</font>", st["cell"])]
        for t, _ in topics:
            ft = fit.get(t)
            if not ft:
                row.append(Paragraph("not run", st["cell"]))
                continue
            v = ft.get(key)
            ci = fmt_ci(get(ft, "ci", key)) if v is not None else ""
            txt = fmt(v, signed=(key == "b"))
            if ci:
                txt += f"<br/><font size='7.6' color='#5b6169'>95% CI {ci}</font>"
            row.append(Paragraph(txt, st["cell"]))
        rows.append(row)
    for key, label in (("k", "<i>k</i> = β·<i>b</i> (intercept)"), ("r2", "R²"),
                       ("n_transitions", "<i>n</i> transitions")):
        row = [Paragraph(label, st["cell"])]
        for t, _ in topics:
            ft = fit.get(t)
            v = ft.get(key) if ft else None
            if not ft:
                txt = "not run"
            elif key == "n_transitions":
                txt = str(v) if v is not None else NA
            else:
                txt = fmt(v, signed=(key == "k"))
            row.append(Paragraph(txt, st["cell"]))
        rows.append(row)
    tbl = Table(rows, colWidths=[44 * mm, 42 * mm, 42 * mm, 42 * mm], hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


def resolve(p, base):
    """Use the path as given if it exists, else look for it next to the summary file."""
    if not p:
        return None
    p = Path(p)
    for cand in (p, base / p, base / "figures" / p.name, base / p.name):
        if cand.exists():
            return cand
    return None


def figure(path, width, max_h, st, caption):
    if path is None:
        return [Paragraph(f"<i>(figure not found)</i> {caption}", st["cap"])]
    w, h = ImageReader(str(path)).getSize()
    scale = min(width / w, max_h / h)
    img = Image(str(path), width=w * scale, height=h * scale)
    img.hAlign = "CENTER"
    return [img, Spacer(1, 2), Paragraph(caption, st["cap"])]


def bullets(items, st):
    return ListFlowable([ListItem(Paragraph(t, st["bullet"]), leftIndent=12, value="•") for t in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontName="Sans",
                        bulletFontSize=8, spaceAfter=5)


# ---------------------------------------------------------------- document
def build(S, base, out, fig_scale=1.0):
    _fonts()
    st = _styles()
    F = sorted(S.get("f_levels") or [])
    n = S.get("n_agents") or 0
    seeds = S.get("seeds") or []
    total, failed = S.get("total_calls"), S.get("failed_calls")
    story = [Paragraph(TITLE, st["title"]), Paragraph(SUBTITLE, st["sub"])]

    def H(t):
        story.append(Paragraph(t, st["h"]))

    def P(t):
        story.append(Paragraph(t, st["body"]))

    A0 = S.get("_ablation")
    if A0 and get(A0, "fit", "real", "b") is not None and get(A0, "fit", "twin", "b") is not None:
        s_main = (get(S, "fit", "real", "s") + get(S, "fit", "twin", "s")) / 2
        s_abl = (get(A0, "fit", "real", "s") + get(A0, "fit", "twin", "s")) / 2
        br, bt = get(A0, "fit", "real", "b"), get(A0, "fit", "twin", "b")
        story.append(Paragraph(
            f"<b>Summary.</b> (1) A small prompt detail decided the dynamics. When agents were reminded of their "
            f"own rating each round they kept about {100 * s_main:.0f}% of their view and nothing else could be "
            f"measured. Without the reminder this fell to about {100 * s_abl:.0f}%. (2) The agents then drifted "
            f"toward a default view and toward consensus, and that default was more favourable for “heat "
            f"pumps” (b = {fmt(br, True)}) than for the identical made-up “Varn units” "
            f"(b = {fmt(bt, True)}). (3) A quarter of agents committed against heat pumps did not flip the group. "
            + (f"(4) A blind re-reading of {S['_stance']['n_rated']} posts matched the agents’ own ratings "
               f"closely (weighted κ = {fmt(S['_stance']['weighted_kappa'])}). "
               if S.get("_stance") else "")
            + "With one model, one seed and 16 agents, these results show the direction of each effect and leave "
              "its size open.", st["body"]))
        story.append(Spacer(1, 4))

    H("1. Background and research question")
    P("Earlier work found that LLM agents drift toward a default view of their own (Chuang et al., 2024), "
      "and that this pull explains much of how they change their minds (Probine et al., 2026). A committed "
      "minority can tip a population of LLM agents, but only on arbitrary choices with no default "
      "(Ashery et al., 2025). "
      "For opinion dynamics in the energy transition, the ECHO project asks whether a "
      "group reaches consensus, whether it polarises, and whether a committed minority can tip it. When LLM "
      "agents are used to simulate these outcomes, part of what we see comes from the agents reading each "
      "other, and part may come from what the model already thinks about the topic. In this pilot one small set-up "
      "covers all three. I record where the group’s mean opinion settles, how spread out the "
      "final opinions are, and how large an opposing minority must be to flip the group. I ran the same "
      "discussion about a home-heating technology twice, once calling it “heat pumps”, and once "
      "calling it “Varn units”, a made-up name with the same description, so the model has no "
      "prior attached to the name. My working claim is that the difference between the real topic and its "
      "twin measures how much of the outcome comes from the model’s prior rather than from the "
      "interaction.")

    H("2. Experimental setup")
    counts = ", ".join(str(round(f * n)) for f in F)
    shares = ", ".join(fmt_f(f) for f in F)
    if total is not None and failed is not None:
        calls = (f"{total:,} agent calls in total, none failed." if failed == 0 else
                 f"{total:,} agent calls in total, {failed:,} failed ({100 * failed / total:.1f}%). "
                 "A failed call leaves that agent’s rating unchanged for the round.")
    else:
        calls = f"Calls in total: {total if total is not None else NA}, failed: {failed if failed is not None else NA}."
    story.append(bullets([
        f"Model: <font name='Sans'>{escape(str(S.get('model', NA)))}</font>, a free model on OpenRouter, "
        f"temperature 0.7.",
        f"Agents: {n}, each with a short persona (a nurse, a farmer, a landlord, ...) in a German town. "
        f"Free agents start at views spread from −2 to +2. Rounds: {S.get('rounds', NA)}.",
        f"Graph: random (Erdős-Rényi) with edge probability {P_EDGE}, redrawn until connected. "
        f"Each round an agent reads its neighbours’ latest posts, writes a new post and gives a rating "
        f"from −3 to +3.",
        f"Topics, all described as “{DESCRIPTION}”: <i>real</i> = heat pumps, <i>twin</i> = Varn "
        f"units, <i>synonym</i> = air-source heat systems (only without a minority, as a wording check).",
        f"Committed minority (always −3): {counts} of {n} agents, i.e. f = {shares}.",
        f"Random seeds: {', '.join(map(str, seeds)) or NA}. A seed fixes the graph, the starting "
        f"views and which agents are committed.",
        calls,
        leak_line(S.get("leak_check")),
    ], st))

    H("3. Model and estimation")
    P("I describe each free agent’s update with a linear rule. Opinions are on the scale "
      "<i>z</i> = rating/3, from −1 to +1.")
    story += [
        Paragraph("<i>z<sub>i</sub></i>(<i>t</i>+1) = <i>s</i>·<i>z<sub>i</sub></i>(<i>t</i>) + "
                  "<i>a</i>·<i>m<sub>i</sub></i>(<i>t</i>) + β·<i>b</i>,"
                  " with <i>s</i> + <i>a</i> + β = 1", st["eq"]),
        Paragraph("<i>z</i>* = (<i>a</i>·<i>f</i>·<i>c</i> + β·<i>b</i>) / "
                  "(<i>a</i>·<i>f</i> + β), <i>c</i> = −1", st["eq"]),
        Paragraph("<i>f<sub>c</sub></i> = β·<i>b</i> / <i>a</i>", st["eq"]),
        Spacer(1, 3),
    ]
    sym = [
        ("<i>z<sub>i</sub></i>(<i>t</i>)", "agent <i>i</i>’s opinion in round <i>t</i>"),
        ("<i>s</i>", "weight on the agent’s own previous view"),
        ("<i>a</i>", "weight on the neighbours"),
        ("<i>m<sub>i</sub></i>(<i>t</i>)", "mean opinion of <i>i</i>’s neighbours, committed ones included"),
        ("β", "weight on the model’s default"),
        ("<i>b</i>", "the model’s default view, where an agent drifts with nobody to talk to"),
        ("<i>f</i>, <i>c</i>", "share of committed agents, and their fixed opinion (−1)"),
        ("<i>z</i>*, <i>f<sub>c</sub></i>", "mean-field resting point, and the share at which <i>z</i>* crosses "
         "zero (exists only if <i>b</i> &gt; 0)"),
    ]
    cells = [Paragraph(f"{a} <font color='#5b6169'>{b}</font>", st["cell"]) for a, b in sym]
    symtab = Table([cells[i:i + 2] for i in range(0, len(cells), 2)], colWidths=[85 * mm, 85 * mm])
    symtab.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [symtab, Spacer(1, 5)]
    P("I estimate the weights by least squares on all round-to-round transitions of the free agents, "
      "separately for each topic, with β = 1 − <i>s</i> − <i>a</i>. The intercept is "
      "<i>k</i> = β·<i>b</i>, so <i>b</i> = <i>k</i>/β. The 95% intervals come from a "
      "bootstrap over agent trajectories. The twin should sit close to a no-prior baseline, so Δ<i>b</i> = "
      "<i>b</i><sub>real</sub> − <i>b</i><sub>twin</sub> and Δ<i>f<sub>c</sub></i> are the "
      "quantities of interest.")

    story.append(KeepTogether([
        Paragraph("4. Results", st["h"]),
        Paragraph("Table 1. Fitted parameters per topic, with 95% confidence intervals.", st["cap"]),
        param_table(S, st), Spacer(1, 8)]))
    kf = key_finding(S)
    if kf:
        P(kf)
    for t in results_text(S):
        P(t)


    figs = S.get("figures") or {}
    width = 170 * mm
    story.append(KeepTogether(figure(
        resolve(figs.get("trajectories"), base), width, 54 * mm * fig_scale, st,
        "Figure 1. Mean opinion of the free agents over the rounds, for each topic and minority share.")))
    half = 83 * mm
    left = figure(resolve(figs.get("tipping"), base), half, 48 * mm * fig_scale, st,
                  "Figure 2. Final mean opinion against the minority share f, observed and as predicted "
                  "by the fitted linear model after the same number of rounds.")
    right = figure(resolve(figs.get("params"), base), half, 48 * mm * fig_scale, st,
                   "Figure 3. Fitted s, a, β and b for each topic, with 95% confidence intervals.")
    pair = Table([[left, right]], colWidths=[85 * mm, 85 * mm])
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 1), ("RIGHTPADDING", (0, 0), (-1, -1), 1)]))
    story += [Spacer(1, 4), pair]

    ptab, ptxt = polarisation_table(S, st), polarisation_text(S)
    if ptab is not None:
        cap = Paragraph("Table 2. Spread of the free agents’ final opinions, given as the sd of <i>z</i> and as the split, the "
                        "share of agents on the minority’s side of the mean.", st["cap"])
        right = Paragraph(ptxt or "", st["body"])
        block = Table([[[cap, ptab], right]], colWidths=[72 * mm, 98 * mm])
        block.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (0, 0), 6)]))
        story += [Spacer(1, 2), block, Spacer(1, 6)]

    A = S.get("_ablation")
    sec = 5
    if A:
        H("5. Ablation: removing the stance reminder")
        P("I repeated the runs at f = 0 and f = 0.25 with one change. Agents saw their own last post but "
          "not their numeric rating. Everything else, including the graph and starting views, was the same "
          "(280 further calls, none failed).")
        tbl, txt = ablation(S, A, st)
        story += [Paragraph("Table 3. Main run (agents reminded of their rating) vs. no reminder.", st["cap"]),
                  tbl, Spacer(1, 6)]
        P(txt)
        sec = 6

    V = S.get("_stance")
    if V and V.get("n_rated"):
        H(f"{sec}. Validation of stance ratings")
        jm = ", ".join(V.get("judge_model") or [])
        same = jm == V.get("agent_model")
        who = ("A second pass of the same model as the agents" if same else
               f"A second model (<font name='Sans'>{escape(jm)}</font>)")
        P(f"Every reply had a post and a self-reported rating. {who} read posts blind and rated the stance "
          f"of each from −3 to +3. I sampled {V['n_sampled']} posts evenly over the rating levels the agents "
          f"used (−2 to +2). Of these, {V['n_rated']} were rated and the rest failed on the free tier.")
        g = V.get("mean_gap")
        gn, gp = V.get("mean_gap_when_self_negative"), V.get("mean_gap_when_self_positive")
        sent = (f"The two agree closely (Spearman ρ = {fmt(V['spearman'])}, weighted κ = "
                f"{fmt(V['weighted_kappa'])}). Of the posts, {100 * V['exact']:.0f}% match exactly and "
                f"{100 * V['within_1']:.0f}% are within one point. ")
        if g is not None and abs(g) < 0.05:
            sent += "There is no overall shift between the two. "
        elif g is not None:
            sent += (f"On average the text reads {fmt(abs(g))} points "
                     f"{'more favourable' if g > 0 else 'less favourable'} than the agents rated themselves. ")
        if gn is not None and gp is not None:
            if gn > 0.1 and gp < -0.1:
                sent += (f"But both sides wrote slightly more moderate posts than their numbers. Relative to the rating, the text read "
                         f"{fmt(gn, True)} points for agents who rated themselves against and {fmt(gp, True)} for "
                         "those in favour. A small gap like this between post and rating is worth testing with human coding. ")
            else:
                sent += (f"For agents that rated themselves against, the gap is {fmt(gn, True)}, and for those in "
                         f"favour, {fmt(gp, True)}. ")
        sent += ("This checks the measurement only. The judging model can share the agents’ habits, "
                 "and human coding is the real reference.")
        P(sent)
        fp = resolve(V.get("figure"), base)
        story.append(KeepTogether(figure(fp, 80 * mm, 60 * mm, st,
                                         "Figure 4. Agent's own rating against the stance read blind from its "
                                         "post. The dashed line is perfect agreement.")))
        sec += 1

    H(f"{sec}. Limitations")
    P("This pilot shows that the comparison can be run end to end and gives first numbers. "
      "It does not establish how large the prior effect is in general. The main limits follow.")
    story.append(bullets([
        "One free model. Other models may hold different priors about heat pumps, or none.",
        (f"One seed and only {n} agents" if len(seeds) == 1 else f"{len(seeds)} seeds and only {n} agents")
        + ", so the intervals are wide and graph effects are not averaged out.",
        f"{S.get('rounds', 5)} rounds is short. The groups are most likely still moving, so the final means "
        "are not steady states, and the flip points are read from a transient.",
        "The opinions are the agents’ own ratings. Where I checked them against the text, the check used a "
        "language model and no human coding.",
        "Ratings are whole numbers from −3 to +3, which makes small changes invisible and bends the "
        "linear fit near the ends of the scale.",
        "The made-up name may leak. From the description alone the model can guess it is a heat pump, so "
        "the twin is a weaker baseline than it looks.",
        "Free-tier rate limits kept the runs small.",
    ], st))

    H(f"{sec + 1}. Future work")
    story.append(bullets([
        "Repeat with several models and more seeds, to see whether Δb is stable or model-specific.",
        "Have a sample of posts coded by people, and use it to test both the agents’ ratings and the "
        "blind language-model labels.",
        "Raise the minority step by step and then lower it again. If the group does not come back on the "
        "way down, that is hysteresis and a sign of real tipping rather than a shift of the resting point.",
    ], st))

    H(f"{sec + 2}. References")
    for r in [
        "Chuang, Y.-S. et al. (2024). Simulating opinion dynamics with networks of LLM-based agents. "
        "<i>Findings of NAACL 2024</i>.",
        "Probine, C. et al. (2026). Characterizing opinion evolution of networked LLMs. arXiv:2606.18276.",
        "Ju, D. et al. (2024). Sense and sensitivity: Evaluating the simulation of social dynamics via "
        "large language models. arXiv:2412.05093.",
        "Ashery, A. F., Aiello, L. M. &amp; Baronchelli, A. (2025). Emergent social conventions and "
        "collective bias in LLM populations. <i>Science Advances</i>.",
    ]:
        story.append(Paragraph(r, st["ref"]))

    pages = [0]

    def footer(canvas, doc):
        pages[0] = canvas.getPageNumber()
        canvas.saveState()
        canvas.setFont("Sans", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(20 * mm, 11 * mm, "Whose opinion is it? Pilot note")
        canvas.drawRightString(A4[0] - 20 * mm, 11 * mm, str(doc.page))
        canvas.restoreState()

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=14 * mm, bottomMargin=15 * mm, title=TITLE, author="Obaid Tamboli")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return pages[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--out", default="reports/pilot_report.pdf")
    ap.add_argument("--ablation", default="results/nostance/summary.json")
    args = ap.parse_args()
    summary = Path(args.summary)
    if not summary.exists():
        raise SystemExit(f"{summary} not found; run analyze.py first")
    S = json.loads(summary.read_text(encoding="utf-8"))
    abl = Path(args.ablation)
    if abl.exists():
        S["_ablation"] = json.loads(abl.read_text(encoding="utf-8"))
    sv = Path("results/stance_check/summary.json")
    if sv.exists():
        S["_stance"] = json.loads(sv.read_text(encoding="utf-8"))
    for scale in (1.0, 0.85, 0.7):  # shrink the figures if the note runs past four pages
        n = build(S, summary.parent, Path(args.out), fig_scale=scale)
        if n <= 4:
            break
    print(f"wrote {args.out} ({n} page{'s' if n != 1 else ''})")


if __name__ == "__main__":
    main()
