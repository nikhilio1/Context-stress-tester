"""
ui_charts.py

Charts and result tables for the app. No Streamlit imports, so everything here can be tested alone.

Colour language, used everywhere in the app:
  green      = the model found the fact
  amber      = partly found (some of several facts)
  red        = the model missed it
  deep red,  = the REQUEST FAILED (rate limit, too large, bad key...). This is an error, not a wrong answer.
  hatched
Every box shows the percentage, the number of trials behind it, and how many requests failed.
"""

import html

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

from ui_strip import FAIL, GREEN, INK, PAPER, RUST

AMBER = "#D9A441"
FAIL_EDGE = "#C0644A"
CMAP = LinearSegmentedColormap.from_list("lab", [RUST, AMBER, GREEN])
CMAP.set_bad(PAPER)  # boxes with no data (and the frame behind failed boxes)

HOW_TO_READ = {
    "depth": "How to read it: each box is one setting (text length by where the fact was hidden). "
             "Green = found, red = missed, hatched deep red = the request itself failed. n = trials behind the box.",
    "facts": "How to read it: each box is one setting (text length by number of hidden facts). "
             "The percentage is the share of facts found. Hatched deep red = the request itself failed. n = trials.",
}


def fmt_depth(c):
    c = float(c)
    name = {0.0: "start", 50.0: "middle", 100.0: "end"}.get(c)
    return f"{name}\n{c:g}%" if name else f"{c:g}%"


def fmt_nfacts(c):
    return f"{int(c)} facts"


def _empty_like(df):
    return pd.DataFrame(columns=df.columns)


def _grid(ok, err, index, columns, value):
    """Return rows, cols, mean value, answered count, failed count for every (row, col) cell."""
    err = _empty_like(ok) if err is None else err
    if len(err):
        err = err.dropna(subset=[index, columns])
    rows = sorted(set(ok[index].dropna()) | set(err[index].dropna() if len(err) else []))
    cols = sorted(set(ok[columns].dropna()) | set(err[columns].dropna() if len(err) else []))
    acc = np.full((len(rows), len(cols)), np.nan)
    cnt = np.zeros((len(rows), len(cols)), dtype=int)
    fail = np.zeros((len(rows), len(cols)), dtype=int)
    for a, i in enumerate(rows):
        for b, j in enumerate(cols):
            s = pd.to_numeric(ok.loc[(ok[index] == i) & (ok[columns] == j), value].astype(float), errors="coerce")
            s = s.dropna()
            if len(s):
                acc[a, b], cnt[a, b] = float(s.mean()), len(s)
            if len(err):
                fail[a, b] = int(((err[index] == i) & (err[columns] == j)).sum())
    return rows, cols, acc, cnt, fail


def heatmap_fig(ok, index, columns, value, title, col_fmt, xlabel, errors=None):
    rows, cols, acc, cnt, fail = _grid(ok, errors, index, columns, value)
    nrow, ncol = len(rows), len(cols)
    has_fail = bool(fail.sum())
    if nrow == 0 or ncol == 0:
        fig, ax = plt.subplots(figsize=(5, 1.5))
        fig.patch.set_facecolor(PAPER)
        ax.axis("off")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=INK)
        return fig

    data = np.ma.masked_invalid(acc)
    fig, ax = plt.subplots(figsize=(max(5.4, 1.6 * ncol + 2.6), max(3.0, 1.0 * nrow + 2.3) + (0.4 if has_fail else 0)))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    im = ax.imshow(data, cmap=CMAP, vmin=0, vmax=1, aspect="auto")

    # hatched deep-red boxes where EVERY request failed (nothing was answered)
    for i in range(nrow):
        for j in range(ncol):
            if cnt[i, j] == 0 and fail[i, j] > 0:
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, facecolor=FAIL, edgecolor=FAIL_EDGE,
                                       hatch="///", linewidth=0, zorder=2))

    ax.set_xticks(np.arange(0.5, ncol - 0.5, 1), minor=True)   # gaps between boxes only
    ax.set_yticks(np.arange(0.5, nrow - 0.5, 1), minor=True)
    ax.grid(which="minor", color=PAPER, linewidth=4)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)

    ax.set_xticks(range(ncol))
    ax.set_xticklabels([col_fmt(c) for c in cols], color=INK, fontsize=10)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([f"{int(i):,} tokens" for i in rows], color=INK, fontsize=10)
    ax.set_xlabel(xlabel, color=INK, fontsize=10, labelpad=8)
    ax.set_title(title, loc="left", color=INK, fontsize=13, family="serif", pad=12)

    for i in range(nrow):
        for j in range(ncol):
            n, k = cnt[i, j], fail[i, j]
            if n == 0 and k > 0:
                ax.text(j, i - 0.1, "FAILED", ha="center", va="center", color="white", fontsize=14,
                        fontweight="bold", family="monospace", zorder=3)
                ax.text(j, i + 0.26, f"{k} request{'s' if k != 1 else ''}", ha="center", va="center",
                        color="white", fontsize=8.5, family="monospace", zorder=3)
                continue
            if n == 0:
                ax.text(j, i, "no data", ha="center", va="center", color=INK, fontsize=9, alpha=0.6)
                continue
            v = data[i, j]
            fg = "white" if (v < 0.3 or v >= 0.85) else INK
            dy = -0.17 if k else -0.1
            ax.text(j, i + dy, f"{v:.0%}", ha="center", va="center", color=fg,
                    fontsize=15, fontweight="bold", family="monospace")
            ax.text(j, i + dy + 0.31, f"n={n}", ha="center", va="center", color=fg, fontsize=8.5, family="monospace")
            if k:
                ax.text(j, i + dy + 0.5, f"+{k} failed", ha="center", va="center", color=fg,
                        fontsize=8.5, family="monospace", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.22, ticks=[0, 0.5, 1])
    cbar.ax.set_xticklabels(["0%  missed", "50%", "100%  found"], color=INK, fontsize=9)
    for lab, ha in zip(cbar.ax.get_xticklabels(), ["left", "center", "right"]):
        lab.set_ha(ha)  # keep the end labels inside the picture
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)

    if has_fail:
        fig.legend(handles=[Patch(facecolor=FAIL, edgecolor=FAIL_EDGE, hatch="///",
                                  label="request failed (an error, not a wrong answer)")],
                   loc="lower left", frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.02, 0.0))
        fig.tight_layout(rect=[0, 0.07, 1, 1])
    else:
        fig.tight_layout()
    return fig


def summary_text(ok, index, columns, value, col_fmt, errors=None):
    """One plain-English sentence about the result."""
    rows, cols, acc, cnt, fail = _grid(ok, errors, index, columns, value)
    total_fail = int(fail.sum())
    n_ok = int(cnt.sum())
    fail_part = ""
    if total_fail:
        fail_part = f" {total_fail} request{'s' if total_fail != 1 else ''} failed (hatched red); the reason is below."
    if n_ok == 0:
        return ("Every request failed, so there is no accuracy to show." + fail_part) if total_fail else ""
    k = int(np.nanargmin(acc))
    i, j = divmod(k, acc.shape[1])
    worst = acc[i, j]
    if worst >= 0.999:
        head = "Every answered setting scored 100%: the model found every hidden fact."
    else:
        label = col_fmt(cols[j]).replace("\n", " ")
        head = f"Weakest answered setting: {int(rows[i]):,} tokens, {label} ({worst:.0%} found)."
    return head + fail_part


def failure_html(err, where_col, where_fmt=lambda v: str(v)):
    """
    A red callout that says WHY requests failed. `err` needs an 'error_summary' column.
    Groups identical reasons and lists where they happened. Returns '' when nothing failed.
    """
    if err is None or len(err) == 0:
        return ""
    items = ""
    for reason, grp in err.groupby("error_summary", dropna=False):
        where = sorted(set(grp[where_col].dropna()))
        where_txt = ", ".join(html.escape(where_fmt(w)) for w in where)
        n = len(grp)
        items += (f"<li><b>{n} request{'s' if n != 1 else ''}</b>"
                  f"{' at ' + where_txt if where_txt else ''}: {html.escape(str(reason))}</li>")
    return (f'<div class="fail"><b>Why requests failed</b><ul>{items}</ul>'
            f'<div class="why">A failed request is an error, not the model giving a wrong answer, '
            f'so it is kept out of the accuracy numbers.</div></div>')


def verdict_html(rows):
    """Tab 2 result table: green FOUND, red NOT FOUND, deep-red FAILED with the reason."""
    head = "<tr><th>Depth</th><th>Model</th><th>Result</th><th>Reply or reason</th></tr>"
    body = ""
    for r in rows:
        res = r["result"]
        cls = {"FOUND": "ok", "NOT FOUND": "miss"}.get(res, "fail")
        label = "FAILED" if res == "ERROR" else res
        text = r.get("error_summary") if res == "ERROR" else r.get("reply", "")
        body += (f'<tr><td>{r["depth_percent"]}%</td><td>{html.escape(str(r["model"]))}</td>'
                 f'<td class="v {cls}">{label}</td><td>{html.escape(str(text or ""))}</td></tr>')
    return f'<table class="vt">{head}{body}</table>'


def doc_heatmap_fig(rows):
    """Tab 2: models down the side, depths across the top. Green = found, red = not found, hatched deep red = failed."""
    models = list(dict.fromkeys(r["model"] for r in rows))
    depths = sorted({float(r["depth_percent"]) for r in rows})
    look = {(r["model"], float(r["depth_percent"])): r["result"] for r in rows}
    nrow, ncol = len(models), len(depths)
    fig, ax = plt.subplots(figsize=(max(5.4, 1.7 * ncol + 3.4), max(2.4, 1.0 * nrow + 2.0)))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for i, m in enumerate(models):
        for j, d in enumerate(depths):
            res = look.get((m, d))
            if res is None:
                ax.text(j, i, "not run", ha="center", va="center", color=INK, fontsize=9, alpha=0.6)
                continue
            if res == "ERROR":
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, facecolor=FAIL, edgecolor=FAIL_EDGE,
                                       hatch="///", linewidth=0))
                label = "FAILED"
            else:
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, facecolor=GREEN if res == "FOUND" else RUST,
                                       linewidth=0))
                label = res
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=13, fontweight="bold",
                    family="monospace", zorder=3)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([fmt_depth(d) for d in depths], color=INK, fontsize=10)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(models, color=INK, fontsize=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Depth where your fact was hidden (each depth is a separate request)", color=INK, fontsize=10, labelpad=8)
    ax.set_title("Did the model find your fact?", loc="left", color=INK, fontsize=13, family="serif", pad=12)
    fig.legend(handles=[Patch(facecolor=GREEN, label="found"), Patch(facecolor=RUST, label="not found (the model answered)"),
                        Patch(facecolor=FAIL, edgecolor=FAIL_EDGE, hatch="///", label="request failed (an error)")],
               loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    return fig


# ====================================================================== Tab 2, mode A: facts already in the document
def qa_heatmap_fig(rows):
    """Models down the side, one column per question (ordered by where the answer sits in the document)."""
    qs = sorted({(r["position"], r["qnum"], r["question"]) for r in rows if r.get("kind") == "question"})
    models = list(dict.fromkeys(r["model"] for r in rows))
    look = {(r["model"], r["qnum"]): r["result"] for r in rows if r.get("kind") == "question"}
    nrow, ncol = len(models), len(qs)
    fig, ax = plt.subplots(figsize=(max(5.4, 1.9 * ncol + 3.4), max(2.4, 1.0 * nrow + 2.1)))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for i, m in enumerate(models):
        for j, (_, qn, _) in enumerate(qs):
            res = look.get((m, qn))
            if res is None:
                ax.text(j, i, "not run", ha="center", va="center", color=INK, fontsize=9, alpha=0.6)
                continue
            if res == "ERROR":
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, facecolor=FAIL, edgecolor=FAIL_EDGE,
                                       hatch="///", linewidth=0))
                label = "FAILED"
            else:
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                       facecolor=GREEN if res == "FOUND" else RUST, linewidth=0))
                label = res
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=12, fontweight="bold",
                    family="monospace", zorder=3)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"Q{qn}\nanswer at {pos:g}%" for pos, qn, _ in qs], color=INK, fontsize=9)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(models, color=INK, fontsize=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Each question's answer already sits in your document at this position", color=INK, fontsize=10, labelpad=8)
    ax.set_title("Did the model find facts that are really in your document?", loc="left", color=INK, fontsize=13,
                 family="serif", pad=12)
    fig.legend(handles=[Patch(facecolor=GREEN, label="found"), Patch(facecolor=RUST, label="not found / wrong answer"),
                        Patch(facecolor=FAIL, edgecolor=FAIL_EDGE, hatch="///", label="request failed (an error)")],
               loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    return fig


def qa_table_html(rows):
    head = "<tr><th>Answer at</th><th>Model</th><th>Question</th><th>Expected</th><th>Result</th><th>Reply or reason</th></tr>"
    body = ""
    for r in sorted((x for x in rows if x.get("kind") == "question"), key=lambda x: (x["position"], x["model"])):
        res = r["result"]
        cls = {"FOUND": "ok", "NOT FOUND": "miss"}.get(res, "fail")
        text = r.get("error_summary") if res == "ERROR" else r.get("reply", "")
        body += (f'<tr><td>{r["position"]:g}%</td><td>{html.escape(str(r["model"]))}</td>'
                 f'<td>{html.escape(r["question"])}</td><td>{html.escape(r["expected"])}</td>'
                 f'<td class="v {cls}">{"FAILED" if res == "ERROR" else res}</td>'
                 f'<td>{html.escape(str(text or ""))}</td></tr>')
    return f'<table class="vt">{head}{body}</table>'


def control_html(rows):
    """One line per model for the control question (a question the document cannot answer)."""
    out = ""
    for r in (x for x in rows if x.get("kind") == "control"):
        res = r["result"]
        cls = {"ABSTAINED": "ok", "MADE UP": "miss"}.get(res, "fail")
        why = {"ABSTAINED": "correctly said NOT FOUND", "MADE UP": "answered something that is not in the document",
               "ERROR": r.get("error_summary") or "request failed"}[res]
        out += (f'<tr><td>{html.escape(str(r["model"]))}</td><td class="v {cls}">{"FAILED" if res == "ERROR" else res}</td>'
                f'<td>{html.escape(why)}</td></tr>')
    return f'<table class="vt"><tr><th>Model</th><th>Control</th><th>What happened</th></tr>{out}</table>' if out else ""