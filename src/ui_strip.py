"""
ui_strip.py

Look and feel for the app, with no Streamlit imports so it can be tested alone.

- CSS: the "lab notebook" theme.
- strip_html(): the "document strip". A long text seen edge-on as a row of ticks,
  with a marker wherever a fact is hidden. While a request runs, a scan line
  sweeps across it. When the answer arrives, each marker turns found or missed.
  It is plain HTML and CSS (no JavaScript), so it renders anywhere markdown does.
"""

import html

PAPER, PAPER2, INK, RUST = "#F6F0E4", "#EFE6D2", "#1F1B16", "#B5472A"

_CSS = """
.block-container { max-width: 1100px; padding-top: 2.2rem; }
footer { visibility: hidden; }
h1, h2, h3 { font-family: Georgia, "Iowan Old Style", serif; font-weight: 600; letter-spacing: -0.01em; }
h1 { font-size: 2.1rem; line-height: 1.15; margin-bottom: .2rem; }
code, pre, [data-testid="stMetricValue"] { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; }
[data-testid="stMetcLabel"] { text-transform: uppercase; letter-spacing: .06em; font-size: .72rem; opacity: .7; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(31,27,22,.35); }
.stButton > button { border-radius: 2px; border: 1px solid __INK__; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: __RUST__; border-color: __RUST__; color: __PAPER__; }
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="textarea"] { border-radius: 2px !important; }
hr { border-top: 1px solid __INK__; opacity: .3; }

.kicker { font: 11px ui-monospace, "Cascadia Mono", Consolas, monospace; letter-spacing: .1em;
          text-transform: uppercase; opacity: .65; margin: 1.4rem 0 .3rem 0; }
.note { border-left: 3px solid __RUST__; padding: .2rem 0 .2rem .8rem; margin: .5rem 0 1rem 0; }
.sub { font-size: 1.02rem; opacity: .8; margin: 0 0 .6rem 0; }

.strip { position: relative; height: 74px; margin: .5rem 0 .2rem 0; overflow: hidden;
         border-top: 1px solid __INK__; border-bottom: 1px solid __INK__; background: __PAPER2__; }
.strip .ticks { position: absolute; inset: 0; display: flex; align-items: center;
                justify-content: space-between; padding: 0 3px; }
.strip .tick { display: block; width: 2px; background: __INK__; opacity: .28; }
.strip .sweep { position: absolute; top: 0; bottom: 0; width: 9%; left: -9%;
                background: rgba(181,71,42,.14); border-right: 2px solid __RUST__;
                animation: strip-sweep 2.4s linear infinite; }
@keyframes strip-sweep { from { left: -9%; } to { left: 100%; } }
.strip .mk { position: absolute; top: 0; bottom: 0; width: 0; border-left: 2px dashed __RUST__; }
.strip .mk.found { border-left: 3px solid __RUST__; }
.strip .mk.missed { border-left: 3px solid __INK__; }
.strip .mk.error { border-left: 2px dotted __INK__; }
.strip .mk-label { position: absolute; top: 4px; left: 6px; padding: 0 4px; background: __PAPER__;
                   font: 11px ui-monospace, "Cascadia Mono", Consolas, monospace; color: __INK__; white-space: nowrap; }
.strip .mk.r .mk-label { left: auto; right: 6px; }
.strip .mk-label:empty { display: none; }
.strip .mk.found .mk-label::after { content: " found"; color: __RUST__; }
.strip .mk.missed .mk-label::after { content: " missed"; }
.strip .mk.error .mk-label::after { content: " error"; }
.strip-cap { font: 12px ui-monospace, "Cascadia Mono", Consolas, monospace; opacity: .7; margin-bottom: .6rem; }
@media (prefers-reduced-motion: reduce) { .strip .sweep { animation: none; left: 0; } }
"""

CSS = (_CSS.replace("__INK__", INK).replace("__PAPER2__", PAPER2)
       .replace("__PAPER__", PAPER).replace("__RUST__", RUST))


def _tick_heights(n=110):
    """Deterministic, irregular heights so the strip reads as lines of text."""
    return [6 + ((i * 37 + (i * i) % 11 * 13 + (i % 7) * 5) % 26) for i in range(n)]


def strip_html(markers=None, state="idle", caption=""):
    """
    markers: list of {"pos": 0-100, "status": pending|found|missed|error, "label": str}
    state:   idle | scanning | done | error  (only 'scanning' shows the sweep)
    Returns one line of HTML (no blank lines, so markdown never splits it).
    """
    ticks = "".join(f'<i class="tick" style="height:{h}px"></i>' for h in _tick_heights())
    marks = ""
    for m in markers or []:
        pos = min(max(float(m["pos"]), 0.0), 100.0)
        status = m.get("status", "pending")
        if status not in ("pending", "found", "missed", "error"):
            status = "pending"
        label = html.escape(str(m.get("label", "")))
        left = "calc(100% - 3px)" if pos >= 99.5 else f"{pos:.1f}%"   # keep the end marker inside the strip
        side = " r" if pos > 80 else ""                               # label sits left of the line near the end
        marks += f'<div class="mk {status}{side}" style="left:{left}"><span class="mk-label">{label}</span></div>'
    sweep = '<div class="sweep"></div>' if state == "scanning" else ""
    cap = f'<div class="strip-cap">{html.escape(caption)}</div>' if caption else ""
    safe_state = state if state in ("idle", "scanning", "done", "error") else "idle"
    return f'<div class="strip {safe_state}"><div class="ticks">{ticks}</div>{sweep}{marks}</div>{cap}'
