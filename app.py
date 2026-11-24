"""
app.py -- Context Window Retrieval Stress Tester (Streamlit)

Run from the project root:
    python -m streamlit run app.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import app_core as core  # noqa: E402
from ui_strip import CSS, INK, PAPER, RUST, strip_html  # noqa: E402

st.set_page_config(page_title="Context Window Stress Tester", layout="wide")

CMAP = LinearSegmentedColormap.from_list("lab", [RUST, "#E9D9BC", "#F6F0E4"])  # low = rust, high = pale

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def heatmap_fig(sub, index, columns, value, title, col_fmt):
    """Mean of `value` per (index, columns) cell, annotated with the sample size."""
    acc = sub.pivot_table(index=index, columns=columns, values=value, aggfunc="mean").sort_index()
    acc = acc.reindex(sorted(acc.columns), axis=1)
    cnt = sub.pivot_table(index=index, columns=columns, values=value, aggfunc="count").reindex_like(acc)

    fig, ax = plt.subplots(figsize=(max(4.5, 1.4 * acc.shape[1] + 1.8), max(2.2, 0.8 * acc.shape[0] + 1.4)))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.imshow(acc.values, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(acc.shape[1]))
    ax.set_xticklabels([col_fmt(c) for c in acc.columns], color=INK, family="monospace", fontsize=9)
    ax.set_yticks(range(acc.shape[0]))
    ax.set_yticklabels([f"{int(i):,}" for i in acc.index], color=INK, family="monospace", fontsize=9)
    ax.set_ylabel("text length (tokens)", color=INK, fontsize=9)
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            v, n = acc.values[i, j], cnt.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0%}\nn={int(n)}", ha="center", va="center",
                        color=PAPER if v < 0.4 else INK, family="monospace", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=11, family="serif")
    fig.tight_layout()
    return fig


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def kicker(text):
    st.markdown(f'<div class="kicker">{text}</div>', unsafe_allow_html=True)


def preview_markers(test, params):
    """Where the facts WILL be hidden, from the current settings."""
    if test == "single":
        return [{"pos": d, "status": "pending", "label": f"{d}%"} for d in params]
    n = params[0]
    return [{"pos": (i + 0.5) / n * 100, "status": "pending", "label": str(i + 1) if n <= 10 else ""}
            for i in range(n)]


def case_markers(job, row=None):
    """Markers for one request. Before the answer: pending. After: found / missed / error."""
    if job["test"] == "single":
        pos, labels = [job["param"]], [f"{job['param']}%"]
    else:
        n = job["param"]
        pos = [(i + 0.5) / n * 100 for i in range(n)]
        labels = [str(i + 1) if n <= 10 else "" for i in range(n)]
    if row is None:
        status = ["pending"] * len(pos)
    elif row["status"] == "ERROR":
        status = ["error"] * len(pos)
    else:
        status = ["found" if f == "1" else "missed" for f in row["found_flags"].split(",")]
    return [{"pos": p, "status": s, "label": l} for p, s, l in zip(pos, status, labels)]


def resolve_keys(models, typed, use_env):
    """Return ({model: key or None}, [providers missing a key])."""
    keys, missing = {}, set()
    for m in models:
        prov = core.provider_of(m)
        if use_env:
            keys[m] = None  # model_clients falls back to .env (local use only)
        elif typed.get(prov, "").strip():
            keys[m] = typed[prov].strip()
        else:
            missing.add(prov)
    return keys, sorted(missing)


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
all_models = list(core.MODEL_REGISTRY.keys())
default_models = [m for m in ["gemini-3.5-flash-lite"] if m in all_models]
models = st.sidebar.multiselect("Models to test", all_models, default=default_models,
                                help="Pick two or three to compare them on the same tests.")

LOCAL = (ROOT / ".env").exists()
use_env = False
if LOCAL:
    use_env = st.sidebar.checkbox("Use keys from my .env file", value=True,
                                  help="Only shown because a .env file exists next to the app (local use).")

typed_keys = {}
if not use_env:
    st.sidebar.markdown("**API keys**")
    for prov in sorted({core.provider_of(m) for m in models}):
        typed_keys[prov] = st.sidebar.text_input(f"{prov} key", type="password", key=f"key_{prov}")
    st.sidebar.caption("Kept in this browser session only. Never saved, logged or written to a file.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Each run is capped at {core.MAX_REQUESTS} requests, so a free quota is not used up by accident.")

# ---------------------------------------------------------------- header
kicker("Evaluation harness")
st.title("Context Window Retrieval Stress Tester")
st.markdown(
    '<p class="sub">Hide a fact in a long text, ask a model to find it, and see where it fails.</p>'
    '<div class="note">Finding one fact is easy for strong models. Finding <b>many at once</b> is much harder, '
    "so both tests are here.</div>",
    unsafe_allow_html=True,
)

tab_live, tab_doc, tab_saved = st.tabs(["Run a test", "Your own document", "Saved results"])

# ================================================================ TAB 1: live test
with tab_live:
    avail = core.corpus_tokens()
    length_opts = [l for l in core.LENGTH_OPTIONS if l <= avail]

    kicker("01 · Configure")
    c1, c2 = st.columns([3, 2])
    test_label = c1.radio("What to hide", ["One fact", "Many facts at once"], horizontal=True)
    trials = c2.slider("Trials per setting", 1, 5, 2,
                       help="More trials make a result more trustworthy but cost more requests.")
    test = "single" if test_label == "One fact" else "multi"

    lengths = st.multiselect("Text length (tokens)", length_opts,
                             default=[10000] if 10000 in length_opts else length_opts[:1],
                             format_func=lambda x: f"{x:,}")
    if test == "single":
        params = st.multiselect("Hide the fact at depth (%)", [0, 25, 50, 75, 100], default=[0, 50, 100],
                                help="0 = very start of the text, 100 = very end.")
    else:
        params = st.multiselect("How many facts to hide", [3, 5, 10, 20], default=[5, 10],
                                help="Facts are spread evenly through the text. The model must report all of them.")

    jobs = core.make_jobs(test, models, lengths, params, trials) if (models and lengths and params) else []
    n_req, n_tok, n_sec = core.estimate(jobs)

    blocked = False
    if not jobs:
        st.info("Choose at least one model, one length and one setting.")
        blocked = True
    else:
        st.markdown(
            f"**Plan:** `{n_req}` requests, about `{n_tok:,}` tokens sent, roughly `{max(1, round(n_sec / 60))}` min "
            "(longer if the provider rate-limits you)."
        )
        if n_req > core.MAX_REQUESTS:
            st.error(f"That is {n_req} requests. The cap is {core.MAX_REQUESTS} per run. Reduce trials, lengths or models.")
            blocked = True
        for m in models:
            prov = core.provider_of(m)
            lim = core.SAFE_TOKEN_LIMIT.get(prov)
            if lim and any(l > lim for l in lengths):
                st.warning(f"{prov} free tier allows only about {lim:,} tokens per request. "
                           f"Longer texts for {m} will fail with rate-limit errors.")

    kicker("02 · Preview and run")
    strip_ph = st.empty()
    if params:
        strip_ph.markdown(
            strip_html(preview_markers(test, params), "idle", "Where the hidden facts will sit in the text"),
            unsafe_allow_html=True,
        )

    run = st.button("Run test", type="primary", disabled=blocked, key="run_live")

    progress_ph = st.empty()
    status_ph = st.empty()
    log_ph = st.empty()
    table_ph = st.empty()

    if run:
        keys, missing = resolve_keys(models, typed_keys, use_env)
        if missing:
            st.error("Enter an API key in the sidebar for: " + ", ".join(missing))
        else:
            rows, logs = [], []
            st.session_state["live_rows"] = rows

            def log(msg):
                logs.append(msg)
                log_ph.code("\n".join(logs[-8:]), language=None)

            bar = progress_ph.progress(0.0)
            haystacks = {}
            for i, job in enumerate(jobs):
                what = f"depth {job['param']}%" if test == "single" else f"{job['param']} facts"
                status_ph.markdown(f"Request {i + 1} of {len(jobs)}: `{job['model']}`, "
                                   f"{job['length']:,} tokens, {what}, trial {job['trial']}")
                if job["length"] not in haystacks:
                    haystacks[job["length"]] = core.build_haystack(job["length"])
                caption = f"{job['model']} · {job['length']:,} tokens · trial {job['trial']}"
                strip_ph.markdown(strip_html(case_markers(job), "scanning", caption + " · searching"),
                                  unsafe_allow_html=True)
                row = core.run_job(job, haystacks[job["length"]], keys[job["model"]], log=log)
                rows.append(row)
                strip_ph.markdown(
                    strip_html(case_markers(job, row), "error" if row["status"] == "ERROR" else "done", caption),
                    unsafe_allow_html=True)
                bar.progress((i + 1) / len(jobs))
                table_ph.dataframe(pd.DataFrame(rows).drop(columns=["auth_error"]), hide_index=True)

                if row["auth_error"]:
                    st.error("The provider rejected the API key. Check it and run again.")
                    break
                if i < len(jobs) - 1 and row["status"] == "OK":
                    wait = core.pause_seconds(row["prompt_tokens"] or 0)
                    log(f"pausing {wait}s to stay under the rate limit")
                    time.sleep(wait)
            status_ph.markdown("Run finished.")

    # ---- results (from this session, so they survive a page refresh of the tab)
    rows = st.session_state.get("live_rows")
    if rows:
        st.markdown("---")
        kicker("03 · Results")
        st.subheader("Results")
        df = pd.DataFrame(rows)
        ok, err = df[df.status == "OK"], df[df.status == "ERROR"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Requests answered", len(ok))
        m2.metric("Mean recall", f"{ok['recall'].mean():.0%}" if len(ok) else "n/a")
        m3.metric("Errors (not counted as wrong)", len(err))
        st.caption("Recall = share of hidden facts the model found. A failed request is an error, never a wrong answer.")

        for (tst, model), sub in ok.groupby(["test", "model"]):
            if tst == "single":
                show_fig(heatmap_fig(sub, "length_tokens", "depth_percent", "recall",
                                     f"{model}: one fact, by depth", lambda c: f"{c:g}%"))
            else:
                show_fig(heatmap_fig(sub, "length_tokens", "n_facts", "recall",
                                     f"{model}: many facts, share found", lambda c: f"{int(c)} facts"))
        if len(err):
            with st.expander("Errors"):
                st.dataframe(err[["model", "length_tokens", "depth_percent", "n_facts", "trial", "error_message"]],
                             hide_index=True)
        with st.expander("Every answer"):
            st.dataframe(df.drop(columns=["auth_error"]), hide_index=True)
        st.download_button("Download results (CSV)", df.drop(columns=["auth_error"]).to_csv(index=False),
                           "live_results.csv", "text/csv")

# ================================================================ TAB 2: own document
with tab_doc:
    st.subheader("Does the model find my fact in my document?")
    st.markdown(
        '<div class="note">This tests the <b>model</b>, not your document. It hides a sentence you write inside '
        "your text and checks whether the model can retrieve it. It does not discover facts on its own. "
        "Your text is sent to the provider you select, so do not paste anything confidential.</div>",
        unsafe_allow_html=True,
    )
    up = st.file_uploader("Upload a .txt file (or paste below)", type=["txt"])
    pasted = st.text_area("Document text", height=200, placeholder="Paste a long document here...")
    document = up.read().decode("utf-8", errors="ignore") if up else pasted

    d1, d2 = st.columns(2)
    fact = d1.text_input("Fact to hide", placeholder="The maintenance key for Hangar 9 is OSPREY-42.")
    question = d2.text_input("Question to ask", placeholder="What is the maintenance key for Hangar 9?")
    expected = st.text_input("Expected answer", placeholder="OSPREY-42")
    doc_depths = st.multiselect("Hide it at depth (%)", [0, 25, 50, 75, 100], default=[0, 50, 100], key="doc_depths")

    doc_tokens = 0
    if document:
        doc_tokens = len(core.multi_needle._ENC.encode(document))
        st.caption(f"Document length: about {doc_tokens:,} tokens.")
    doc_jobs = len(models) * len(doc_depths)
    ready = bool(document and fact and question and expected and models and doc_depths)
    too_many = doc_jobs > core.MAX_REQUESTS
    if too_many:
        st.error(f"That is {doc_jobs} requests. The cap is {core.MAX_REQUESTS} per run.")
    for m in models:
        lim = core.SAFE_TOKEN_LIMIT.get(core.provider_of(m))
        if lim and doc_tokens > lim:
            st.warning(f"{core.provider_of(m)} free tier allows only about {lim:,} tokens per request.")

    doc_strip_ph = st.empty()
    if doc_depths:
        doc_strip_ph.markdown(
            strip_html([{"pos": d, "status": "pending", "label": f"{d}%"} for d in doc_depths], "idle",
                       "Where your fact will be hidden in your document"),
            unsafe_allow_html=True)

    run_doc = st.button("Test my document", type="primary", disabled=not ready or too_many, key="run_doc")
    if run_doc:
        keys, missing = resolve_keys(models, typed_keys, use_env)
        if missing:
            st.error("Enter an API key in the sidebar for: " + ", ".join(missing))
        else:
            doc_rows = []
            st.session_state["doc_rows"] = doc_rows
            doc_log_ph, doc_table_ph = st.empty(), st.empty()
            todo = [(m, d) for m in models for d in doc_depths]
            bar = st.progress(0.0)
            logs = []

            def dlog(msg):
                logs.append(msg)
                doc_log_ph.code("\n".join(logs[-6:]), language=None)

            for i, (m, d) in enumerate(todo):
                one = [{"pos": d, "status": "pending", "label": f"{d}%"}]
                doc_strip_ph.markdown(strip_html(one, "scanning", f"{m} · searching"), unsafe_allow_html=True)
                row = core.run_own_document(m, document, fact, question, expected, d, keys[m], log=dlog)
                doc_rows.append(row)
                one[0]["status"] = {"FOUND": "found", "NOT FOUND": "missed"}.get(row["result"], "error")
                doc_strip_ph.markdown(
                    strip_html(one, "error" if row["result"] == "ERROR" else "done", f"{m} · depth {d}%"),
                    unsafe_allow_html=True)
                bar.progress((i + 1) / len(todo))
                doc_table_ph.dataframe(pd.DataFrame(doc_rows).drop(columns=["auth_error"]), hide_index=True)
                if row["auth_error"]:
                    st.error("The provider rejected the API key. Check it and run again.")
                    break
                if i < len(todo) - 1 and row["result"] != "ERROR":
                    wait = core.pause_seconds(row["prompt_tokens"] or 0)
                    dlog(f"pausing {wait}s to stay under the rate limit")
                    time.sleep(wait)

    doc_rows = st.session_state.get("doc_rows")
    if doc_rows:
        st.markdown("---")
        st.subheader("Verdict by depth")
        ddf = pd.DataFrame(doc_rows).drop(columns=["auth_error"])
        st.dataframe(ddf, hide_index=True)
        n_found = int((ddf.result == "FOUND").sum())
        n_ok = int((ddf.result != "ERROR").sum())
        st.markdown(f"Found the fact in `{n_found}` of `{n_ok}` answered requests.")
        st.caption("FOUND means your expected answer appears in the reply (case-insensitive match).")

# ================================================================ TAB 3: saved results
with tab_saved:
    st.subheader("Saved results")
    st.caption("Read from results/results.csv (single fact) and results/multi_results.csv (many facts).")

    single_p, multi_p = ROOT / "results" / "results.csv", ROOT / "results" / "multi_results.csv"

    if single_p.exists():
        s = pd.read_csv(single_p)
        if "status" not in s.columns:
            s["status"] = "OK"
        s = s[s["status"] == "OK"].copy()
        s["is_correct"] = s["correct"].astype(str).str.lower() == "true"
        s = s.drop_duplicates(["model", "length_tokens", "depth_percent", "trial"], keep="last")
        st.markdown("#### One fact")
        pick = st.selectbox("Model", sorted(s["model"].unique()), key="saved_single_model")
        sub = s[s["model"] == pick]
        k1, k2 = st.columns(2)
        k1.metric("Trials", len(sub))
        k2.metric("Accuracy", f"{sub['is_correct'].mean():.0%}")
        show_fig(heatmap_fig(sub, "length_tokens", "depth_percent", "is_correct",
                             f"{pick}: accuracy by length and depth", lambda c: f"{c:g}%"))
        if "needle_type" in sub.columns:
            acc = sub.pivot_table(index="length_tokens", columns="needle_type", values="is_correct", aggfunc="mean")
            cnt = sub.pivot_table(index="length_tokens", columns="needle_type", values="is_correct", aggfunc="count")
            combo = acc.astype(object)
            for i in acc.index:
                for j in acc.columns:
                    if pd.isna(acc.loc[i, j]):
                        combo.loc[i, j] = "-"
                    else:
                        combo.loc[i, j] = f"{acc.loc[i, j]:.0%} (n={int(cnt.loc[i, j])})"
            st.markdown("Accuracy by needle type")
            st.dataframe(combo)
        miss_cols = [c for c in ["length_tokens", "depth_percent", "trial", "needle_type",
                                 "expected_answer", "model_answer"] if c in sub.columns]
        misses = sub[~sub["is_correct"]][miss_cols]
        st.markdown(f"Misses: `{len(misses)}`")
        if len(misses):
            st.dataframe(misses, hide_index=True)
    else:
        st.info("No results/results.csv yet. Run the command-line harness first.")

    if multi_p.exists():
        mm = pd.read_csv(multi_p)
        mm = mm[mm["status"] == "OK"].drop_duplicates(["model", "length_tokens", "n_facts", "trial"], keep="last")
        st.markdown("---")
        st.markdown("#### Many facts at once")
        if len(mm):
            pick2 = st.selectbox("Model", sorted(mm["model"].unique()), key="saved_multi_model")
            sub2 = mm[mm["model"] == pick2]
            show_fig(heatmap_fig(sub2, "length_tokens", "n_facts", "recall",
                                 f"{pick2}: share of facts found", lambda c: f"{int(c)} facts"))
            st.caption("Within one answer, are facts at the start, middle or end of the text missed more often?")
            n_pick = st.selectbox("Facts hidden", sorted(sub2["n_facts"].unique()), key="saved_multi_n")
            l_pick = st.selectbox("Text length", sorted(sub2["length_tokens"].unique()), key="saved_multi_len")
            cell = sub2[(sub2["n_facts"] == n_pick) & (sub2["length_tokens"] == l_pick)]
            flags = cell["found_flags"].astype(str).str.split(",", expand=True).astype(int)
            by_pos = flags.mean()
            fig, ax = plt.subplots(figsize=(6, 2.4))
            fig.patch.set_facecolor(PAPER)
            ax.set_facecolor(PAPER)
            ax.bar(range(len(by_pos)), by_pos.values, color=RUST)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("position in text (first = nearest the start)", color=INK, fontsize=9)
            ax.set_ylabel("share found", color=INK, fontsize=9)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            fig.tight_layout()
            show_fig(fig)
            st.caption(f"Based on {len(cell)} trial(s).")
    else:
        st.caption("No results/multi_results.csv yet. Run src/multi_needle.py to create it.")