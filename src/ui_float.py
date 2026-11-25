"""
ui_float.py

A background layer of slowly drifting model symbols and project "tokens". No boxes, no borders:
just marks floating on the paper, like a screen full of instruments.

- Eight original abstract glyphs, one per model family, each with its name underneath.
  They are drawn here and are NOT the companies' real logos (those are trademarks).
- If you have the right to use the official logos, drop files into  assets/logos/  named
  claude.svg, gemini.svg, groq.svg, openai.svg, llama.svg, mistral.svg, deepseek.svg, qwen.svg
  (png, webp or jpg also work). A logo file replaces that family's glyph, shown in greyscale.
- Text tokens come from the project itself: the models you selected, a made-up access code,
  NOT FOUND, a 429 rate limit, token counts.
- Plain HTML and CSS, no JavaScript. Behind the page (z-index 0), cannot be clicked,
  hidden on narrow screens, and still for visitors who have reduced motion enabled.
- No Streamlit imports, so it can be tested alone.
"""

import base64
import html
from pathlib import Path

from ui_strip import INK, RUST

# (display name, logo file key, glyph name)
FAMILIES = [
    ("Claude", "claude", "ring"),
    ("Gemini", "gemini", "orbit"),
    ("Groq", "groq", "bars"),
    ("GPT", "openai", "squares"),
    ("Llama", "llama", "layers"),
    ("Mistral", "mistral", "grid"),
    ("DeepSeek", "deepseek", "wave"),
    ("Qwen", "qwen", "target"),
]

# (offset %, y %, kind, size, side). Side slots are anchored to the screen edge so they stay in the margin.
# kinds: g = glyph, tl / tr = short token (left / right margin), tx = long token (free areas only)
_SLOTS = [
    (1.5, 9, "g", 2, "l"), (3, 27, "tl", 1, "l"), (1.5, 45, "g", 3, "l"), (3, 66, "tl", 1, "l"), (1.5, 80, "g", 2, "l"),
    (1.5, 11, "g", 3, "r"), (2, 32, "tr", 1, "r"), (2, 50, "g", 2, "r"), (1.5, 70, "g", 2, "r"), (2, 90, "tr", 1, "r"),
    (64, 6, "g", 1, "l"), (62, 15, "tx", 1, "l"),
    (22, 91, "tx", 1, "l"), (48, 92, "g", 1, "l"), (69, 88, "tx", 1, "l"),
]
_TOK_LEFT = ["429", "n=12"]
_TOK_RIGHT = ["needle", "KR-4821"]
_TOK_LONG = ["NOT FOUND", "100K tokens", "recall 100%", "depth 50%"]
_KEYFRAMES = ["fl-a", "fl-b", "fl-c", "fl-d", "fl-e", "fl-f"]

_PATHS = {
    "ring": '<circle cx="24" cy="24" r="16"/><circle cx="35" cy="13" r="3.8" fill="R" stroke="none"/>',
    "orbit": '<ellipse cx="24" cy="24" rx="20" ry="8.5" transform="rotate(-28 24 24)"/>'
             '<circle cx="24" cy="24" r="3.4" fill="I" stroke="none"/><circle cx="40" cy="15.4" r="3.4" fill="R" stroke="none"/>',
    "bars": '<rect x="7" y="27" width="8" height="14" rx="2"/><rect x="20" y="17" width="8" height="24" rx="2"/>'
            '<rect x="33" y="7" width="8" height="34" rx="2" fill="R" stroke="R"/>',
    "squares": '<rect x="8" y="8" width="32" height="32" rx="3"/>'
               '<rect x="18" y="18" width="12" height="12" transform="rotate(45 24 24)" fill="R" stroke="none"/>',
    "layers": '<path d="M5 17 L24 8 L43 17 L24 26 Z" fill="R" fill-opacity=".28"/><path d="M5 26 L24 35 L43 26"/><path d="M5 35 L24 44 L43 35"/>',
    "grid": ''.join(f'<circle cx="{x}" cy="{y}" r="2.8" fill="I" stroke="none"/>' for x in (11, 24, 37) for y in (11, 24, 37) if (x, y) != (24, 24))
            + '<circle cx="24" cy="24" r="5" fill="R" stroke="none"/>',
    "wave": '<path d="M3 27 C10 8,17 8,23 27 S36 46,45 21"/><circle cx="23" cy="27" r="3.6" fill="R" stroke="none"/>',
    "target": '<circle cx="24" cy="24" r="18"/><circle cx="24" cy="24" r="9"/><path d="M24 2 v9 M24 37 v9 M2 24 h9 M37 24 h9"/>'
              '<circle cx="24" cy="24" r="3" fill="R" stroke="none"/>',
}

_CSS = """
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] { background: transparent !important; }
.block-container, [data-testid="stSidebar"] { position: relative; z-index: 1; }
.fl-layer { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.fl { position: absolute; display: flex; flex-direction: column; align-items: center; gap: 6px;
      animation-timing-function: ease-in-out; animation-iteration-count: infinite; animation-direction: alternate; }
.fl .gl svg, .fl .gl img { display: block; }
.fl .gl svg { fill: none; stroke: __INK__; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; }
.fl .gl img { filter: grayscale(1) contrast(.9); object-fit: contain; }
.fl.g { opacity: .78; }
.fl.s1 .gl svg { width: 34px; height: 34px; } .fl.s1 .gl img { height: 34px; max-width: 80px; }
.fl.s2 .gl svg { width: 50px; height: 50px; } .fl.s2 .gl img { height: 50px; max-width: 110px; }
.fl.s3 .gl svg { width: 72px; height: 72px; } .fl.s3 .gl img { height: 72px; max-width: 150px; }
.fl .lb { font: 11px ui-monospace, "Cascadia Mono", Consolas, monospace; letter-spacing: .08em; text-transform: uppercase; color: __INK__; }
.fl.t { opacity: .55; font: 600 15px ui-monospace, "Cascadia Mono", Consolas, monospace; color: __INK__; letter-spacing: .02em; }
.fl.t::before { content: ""; display: inline-block; width: 7px; height: 7px; background: __RUST__; margin-right: 8px; }
.fl.t { flex-direction: row; gap: 0; align-items: center; white-space: nowrap; }
@keyframes fl-a { from { transform: translate(0, 0) rotate(-3deg); } to { transform: translate(28px, 40px) rotate(3deg); } }
@keyframes fl-b { from { transform: translate(0, 0) rotate(2deg); } to { transform: translate(-34px, 26px) rotate(-4deg); } }
@keyframes fl-c { from { transform: translate(0, 0) rotate(3deg); } to { transform: translate(20px, -42px) rotate(-2deg); } }
@keyframes fl-d { from { transform: translate(0, 0) rotate(-2deg); } to { transform: translate(-26px, -32px) rotate(4deg); } }
@keyframes fl-e { from { transform: translateY(0) rotate(0deg); } to { transform: translateY(-30px) rotate(6deg); } }
@keyframes fl-f { from { transform: translateX(0) rotate(0deg); } to { transform: translateX(34px) rotate(-6deg); } }
@media (max-width: 1400px) { .fl.g { opacity: .65; } .fl.s3 .gl svg { width: 56px; height: 56px; } .fl.s2 .gl svg { width: 40px; height: 40px; } }
@media (max-width: 1240px) { .fl-layer { display: none; } }
@media (prefers-reduced-motion: reduce) { .fl { animation: none !important; } }
"""
CSS = _CSS.replace("__INK__", INK).replace("__RUST__", RUST)


def _svg(name):
    inner = _PATHS[name].replace('"R"', f'"{RUST}"').replace('"I"', f'"{INK}"')
    return f'<svg viewBox="0 0 48 48" aria-hidden="true">{inner}</svg>'


def load_logos(folder):
    """Read optional logo files from `folder` -> {key: data URI}. Missing folder or files are fine."""
    out = {}
    mime = {".svg": "image/svg+xml", ".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    try:
        p = Path(folder)
        if not p.is_dir():
            return out
        for f in p.iterdir():
            if f.suffix.lower() in mime and f.is_file() and f.stat().st_size <= 300_000:
                out[f.stem.lower()] = f"data:{mime[f.suffix.lower()]};base64," + base64.b64encode(f.read_bytes()).decode()
    except OSError:
        pass
    return out


def float_html(models=None, logos=None):
    """One line of HTML for the layer. `models` = labels shown as long text tokens; `logos` = from load_logos()."""
    logos = logos or {}
    fam = iter(FAMILIES)
    pools = {"tl": iter(_TOK_LEFT), "tr": iter(_TOK_RIGHT), "tx": iter(list(models or []) + _TOK_LONG)}
    chips = ""
    for i, (off, y, kind, size, side) in enumerate(_SLOTS):
        anim = _KEYFRAMES[(i * 5) % len(_KEYFRAMES)]
        dur = 26 + (i * 7) % 21           # 26 to 46 seconds: slow, background pace
        delay = -((i * 9) % 23)           # negative delay so chips are already mid-drift on load
        edge = "left" if side == "l" else "right"
        style = f'{edge}:{off}%;top:{y}%;animation-name:{anim};animation-duration:{dur}s;animation-delay:{delay}s'
        if kind == "g":
            name, key, glyph = next(fam)
            body = (f'<img src="{logos[key]}" alt="">' if key in logos else _svg(glyph))
            chips += (f'<div class="fl g s{size}" style="{style}"><div class="gl">{body}</div>'
                      f'<div class="lb">{html.escape(name)}</div></div>')
        else:
            chips += f'<div class="fl t" style="{style}">{html.escape(next(pools[kind]))}</div>'
    return f'<div class="fl-layer" aria-hidden="true">{chips}</div>'