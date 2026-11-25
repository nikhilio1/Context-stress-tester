"""
app.py -- Context Window Retrieval Stress Tester (Streamlit)

Run from the project root:
    python -m streamlit run app.py
"""

import html
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import app_core as core  # noqa: E402
import ui_charts as charts  # noqa: E402
import ui_float  # noqa: E402
from ui_strip import CSS, INK, PAPER, RUST, strip_html  # noqa: E402

st.set_page_config(page_title="Context Window Stress Tester", layout="wide")
st.markdown(f"<style>{CSS}{ui_float.CSS}</style>", unsafe_allow_html=True)

LOGOS = ui_float.load_logos(ROOT / "assets" / "logos")  # optional, user-supplied


# ---------------------------------------------------------------- helpers
def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def show_heatmap(ok, err, index, columns, value, title, kind):
    """kind is 'depth' (one fact) or 'facts' (many facts). `err` = failed requests, shown red and explained."""
    fmt, xlabel = (charts.fmt_depth, "Where the fact was hidden in the text") if kind == "depth" \
        else (charts.fmt_nfacts, "How many facts were hidden")
    show_fig(charts.heatmap_fig(ok, index, columns, value, title, fmt, xlabel, errors=err))
    st.markdown(f"**{charts.summary_text(ok, index, columns, value, fmt, errors=err)}**")
    if err is not None and len(err):
        st.markdown(charts.failure_html(err, "length_tokens", lambda v: f"{int(v):,} tokens"), unsafe_allow_html=True)
    st.caption(charts.HOW_TO_READ[kind])


def multi(r):
    return "" if r["occurrences"] == 1 else f" (appears {r['occurrences']} times)"


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


def saved_stats():
    """Real numbers from results/results.csv for the header strip, or None."""
    p = ROOT / "results" / "results.csv"
    if not p.exists():
        return None
    try:
        s = pd.read_csv(p)
        if "status" not in s.columns:
            s["status"] = "OK"
        ok = s[s["status"] == "OK"].drop_duplicates(["model", "length_tokens", "depth_percent", "trial"], keep="last")
        if not len(ok):
            return None
        correct = ok["correct"].astype(str).str.lower() == "true"
        return {"trials": len(ok), "models": ok["model"].nunique(),
                "longest": int(ok["length_tokens"].max()), "misses": int((~correct).sum())}
    except Exception:
        return None


def split_saved(raw, key):
    """Saved CSV -> (answered rows, failed rows). A failed row is dropped if the same trial later succeeded."""
    if "status" not in raw.columns:
        raw = raw.assign(status="OK")
    ok = raw[raw["status"] == "OK"].drop_duplicates(key, keep="last").copy()
    err = raw[raw["status"] == "ERROR"].drop_duplicates(key, keep="last").copy()
    if len(err) and len(ok):
        err = err[~err.set_index(key).index.isin(ok.set_index(key).index)]
    err["error_summary"] = err["error_message"].fillna("").map(core.explain_error) if len(err) else []
    return ok, err


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
all_models = list(core.MODEL_REGISTRY.keys())
default_models = [m for m in ["gemini-3.5-flash-lite"] if m in all_models]
models = st.sidebar.multiselect("Models to test", all_models, default=default_models,
                                help="Pick two or three to compare them on the same tests.")

st.markdown(ui_float.float_html(models, LOGOS), unsafe_allow_html=True)

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
    "so both tests are here.</div>"
    '<div class="steps">'
    '<div><span class="n">01</span><div class="t">Hide</div><div class="d">A made-up fact goes into a long text, at a depth you choose.</div></div>'
    '<div><span class="n">02</span><div class="t">Ask</div><div class="d">The model is asked to find it, with no randomness (temperature 0).</div></div>'
    '<div><span class="n">03</span><div class="t">Score</div><div class="d">Found, missed, or the request itself failed. Failures are never counted as wrong answers.</div></div>'
    "</div>",
    unsafe_allow_html=True,
)
stats = saved_stats()
if stats:
    st.markdown(
        '<div class="stats">'
        f'<div><div class="num">{stats["trials"]}</div><div class="lab">trials in saved results</div></div>'
        f'<div><div class="num">{stats["longest"]:,}</div><div class="lab">longest text (tokens)</div></div>'
        f'<div><div class="num">{stats["models"]}</div><div class="lab">model{"s" if stats["models"] != 1 else ""} tested</div></div>'
        f'<div><div class="num">{stats["misses"]}</div><div class="lab">misses recorded</div></div>'
        "</div>",
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
                st.warning(f"{prov}'s free tier only allows about {lim:,} tokens at a time. "
                           f"Longer texts for {m} will be rejected. You can still run it: the failures show up in red "
                           "with the reason.")

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

                if row["status"] == "ERROR":
                    log("FAILED: " + row["error_summary"])
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
        m2.metric("Mean recall", f"{ok['recall'].astype(float).mean():.0%}" if len(ok) else "n/a")
        m3.metric("Requests failed", len(err))
        st.caption("Recall = share of hidden facts the model found. A failed request is an error, "
                   "never counted as a wrong answer.")

        for (tst, model), sub in df.groupby(["test", "model"]):
            sub_ok, sub_err = sub[sub.status == "OK"], sub[sub.status == "ERROR"]
            if tst == "single":
                show_heatmap(sub_ok, sub_err, "length_tokens", "depth_percent", "recall",
                             f"{model}: one fact, by depth", "depth")
            else:
                show_heatmap(sub_ok, sub_err, "length_tokens", "n_facts", "recall",
                             f"{model}: many facts, share found", "facts")
        if len(err):
            with st.expander("Raw error messages"):
                st.dataframe(err[["model", "length_tokens", "depth_percent", "n_facts", "trial", "error_message"]],
                             hide_index=True)
        with st.expander("Every answer"):
            st.dataframe(df.drop(columns=["auth_error"]), hide_index=True)
        st.download_button("Download results (CSV)", df.drop(columns=["auth_error"]).to_csv(index=False),
                           "live_results.csv", "text/csv")

# ================================================================ TAB 2: own document
with tab_doc:
    st.subheader("Does the model find facts in my document?")
    st.markdown(
        '<div class="note">Upload your own document. Either ask about facts <b>already in it</b> (the document is sent '
        "exactly as it is), or plant a new sentence to stress-test the model. Your text is sent to the provider you "
        "select, so do not paste anything confidential.</div>",
        unsafe_allow_html=True,
    )
    up = st.file_uploader("Upload a .txt file (or paste below)", type=["txt"])
    pasted = st.text_area("Document text", height=200, placeholder="Paste a long document here...")
    document = up.read().decode("utf-8", errors="ignore") if up else pasted
    doc_tokens = len(core.multi_needle._ENC.encode(document)) if document else 0
    if document:
        st.caption(f"Document length: about {doc_tokens:,} tokens.")

    MODE_A, MODE_B = "Facts already in my document", "Plant a fact (stress test)"
    mode = st.radio("What do you want to test?", [MODE_A, MODE_B], horizontal=True, key="doc_mode")

    if mode == MODE_A:
        st.caption("Nothing is added to your document. For each question the app first checks that the expected answer "
                   "really is in your text, so FOUND can only mean the model read it.")
        qa_text = st.text_area("Questions, one per line, as  question | expected answer", height=120, key="qa_text",
                               placeholder="Who was the founding librarian? | Edmund Thackeray\n"
                                           "How much did the book sale raise? | 7,413")
        items, problems = core.parse_qa(qa_text)
        runnable, skipped = core.check_items(document, items) if document else ([], [])
        for p in problems:
            st.warning(f"Line {p['line']} ignored: {p['why']}.")
        if document and items:
            rows_html = "".join(
                f'<tr><td>{html.escape(r["question"])}</td><td>{html.escape(r["expected"])}</td>'
                f'<td class="v ok">in document at {r["position"]:g}%{multi(r)}</td></tr>'
                for r in runnable) + "".join(
                f'<tr><td>{html.escape(r["question"])}</td><td>{html.escape(r["expected"])}</td>'
                f'<td class="v fail">skipped: {html.escape(r["why"])}</td></tr>' for r in skipped)
            st.markdown('<table class="vt"><tr><th>Question</th><th>Expected</th><th>Check before running</th></tr>'
                        + rows_html + "</table>", unsafe_allow_html=True)
        n_per_model = len(runnable) + 1  # plus one control question
        qa_jobs = len(models) * n_per_model
        qa_too_many = qa_jobs > core.MAX_REQUESTS
        if runnable:
            st.caption(f"{len(runnable)} question(s) plus 1 control question (something your document cannot answer; "
                       f"the model should say NOT FOUND) = {qa_jobs} requests.")
        if qa_too_many:
            st.error(f"That is {qa_jobs} requests. The cap is {core.MAX_REQUESTS} per run.")
        for m in models:
            lim = core.SAFE_TOKEN_LIMIT.get(core.provider_of(m))
            if lim and doc_tokens > lim:
                st.warning(f"{core.provider_of(m)}'s free tier only allows about {lim:,} tokens at a time. "
                           "A text this long will be rejected, and the failure will show in red with the reason.")
        qa_strip_ph = st.empty()
        if runnable:
            qa_strip_ph.markdown(
                strip_html([{"pos": r["position"], "status": "pending", "label": f"Q{r['qnum']}"} for r in runnable],
                           "idle", "Where each answer already sits in your document"), unsafe_allow_html=True)
        run_qa = st.button("Test my document", type="primary", disabled=not (runnable and models) or qa_too_many,
                           key="run_qa")
        if run_qa:
            keys, missing = resolve_keys(models, typed_keys, use_env)
            if missing:
                st.error("Enter an API key in the sidebar for: " + ", ".join(missing))
            else:
                qa_rows = []
                st.session_state["qa_rows"] = qa_rows
                qlog_ph, qtable_ph = st.empty(), st.empty()
                bar, logs, done, stop = st.progress(0.0), [], 0, False

                def qlog(msg):
                    logs.append(msg)
                    qlog_ph.code("\n".join(logs[-6:]), language=None)

                for m in models:
                    marks = [{"pos": r["position"], "status": "pending", "label": f"Q{r['qnum']}"} for r in runnable]
                    for k, it in enumerate(runnable + [None]):
                        if it is not None:
                            qa_strip_ph.markdown(strip_html(marks, "scanning", f"{m} · reading for Q{it['qnum']}"),
                                                 unsafe_allow_html=True)
                            row = core.run_question(m, document, it, keys[m], log=qlog)
                            marks[k]["status"] = {"FOUND": "found", "NOT FOUND": "missed"}.get(row["result"], "error")
                        else:
                            row = core.run_control(m, document, keys[m], log=qlog)
                        qa_rows.append(row)
                        done += 1
                        bar.progress(done / qa_jobs)
                        qtable_ph.markdown(charts.qa_table_html(qa_rows), unsafe_allow_html=True)
                        if row["result"] == "ERROR":
                            qlog("FAILED: " + row["error_summary"])
                        if row.get("auth_error"):
                            st.error("The provider rejected the API key. Check it and run again.")
                            stop = True
                            break
                        if done < qa_jobs and row["result"] != "ERROR":
                            wait = core.pause_seconds(row["prompt_tokens"] or 0)
                            qlog(f"pausing {wait}s to stay under the rate limit")
                            time.sleep(wait)
                    qa_strip_ph.markdown(strip_html(marks, "done", f"{m} · finished"), unsafe_allow_html=True)
                    if stop:
                        break

        qa_rows = st.session_state.get("qa_rows")
        if qa_rows:
            st.markdown("---")
            kicker("Result")
            st.subheader("Did the model find what is really in your document?")
            qdf = pd.DataFrame(qa_rows)
            qq, cc = qdf[qdf.kind == "question"], qdf[qdf.kind == "control"]
            if len(qq):
                show_fig(charts.qa_heatmap_fig(qa_rows))
                st.markdown(charts.qa_table_html(qa_rows), unsafe_allow_html=True)
                st.markdown(f"Found `{int((qq.result == 'FOUND').sum())}` · not found `{int((qq.result == 'NOT FOUND').sum())}` "
                            f"· request failed `{int((qq.result == 'ERROR').sum())}`")
            st.markdown("**Control question** (the document cannot answer it, so the right reply is NOT FOUND)")
            st.markdown(charts.control_html(qa_rows), unsafe_allow_html=True)
            if len(cc) and (cc.result == "MADE UP").any():
                st.warning("A model answered a question your document cannot answer. Treat its FOUND results with care: "
                           "it may be inventing answers.")
            errs = qdf[qdf.result == "ERROR"]
            if len(errs):
                st.markdown(charts.failure_html(errs, "question", lambda v: str(v)[:60]), unsafe_allow_html=True)
            with st.expander("How was this checked?"):
                st.markdown(
                    "- Your document was sent **unchanged**; nothing was inserted.\n"
                    "- Before any request, each expected answer was looked up in your text. Answers that are not there, "
                    "or that are written inside the question, were skipped.\n"
                    "- FOUND needs the expected answer in the reply **and** the reply must not say NOT FOUND.\n"
                    "- The control question asks about something invented. A model that answers it is guessing, "
                    "so its FOUND results are less trustworthy.\n"
                    "- FAILED means the request itself failed. It says nothing about the model's reading.")

    else:
        d1, d2 = st.columns(2)
        fact = d1.text_input("Fact to hide", placeholder="The maintenance key for Hangar 9 is OSPREY-42.")
        question = d2.text_input("Question to ask", placeholder="What is the maintenance key for Hangar 9?")
        expected = st.text_input("Expected answer", placeholder="OSPREY-42")
        doc_depths = st.multiselect("Hide it at depth (%)", [0, 25, 50, 75, 100], default=[0, 50, 100], key="doc_depths")
        st.caption("Each depth is a separate request: your fact is hidden once per request, at that one depth. "
                   "It is never hidden at all the depths at the same time.")

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
                st.warning(f"{core.provider_of(m)}'s free tier only allows about {lim:,} tokens at a time. "
                           "A text this long will be rejected, and the failure will show in red with the reason.")

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
                    doc_table_ph.markdown(charts.verdict_html(doc_rows), unsafe_allow_html=True)
                    if row["result"] == "ERROR":
                        dlog("FAILED: " + row["error_summary"])
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
            kicker("Result")
            st.subheader("Verdict by depth")
            ddf = pd.DataFrame(doc_rows)
            show_fig(charts.doc_heatmap_fig(doc_rows))
            st.markdown(charts.verdict_html(doc_rows), unsafe_allow_html=True)
            n_found = int((ddf.result == "FOUND").sum())
            n_miss = int((ddf.result == "NOT FOUND").sum())
            n_fail = int((ddf.result == "ERROR").sum())
            st.markdown(f"Found `{n_found}` · not found `{n_miss}` · request failed `{n_fail}`")
            if n_fail:
                st.markdown(charts.failure_html(ddf[ddf.result == "ERROR"], "depth_percent", lambda v: f"{v:g}% depth"),
                            unsafe_allow_html=True)
            st.caption("FOUND means your expected answer appears in the reply (case-insensitive match). "
                       "FAILED means the request never got an answer, so it says nothing about the model's reading.")
            with st.expander("Where your fact was placed in each request"):
                st.caption("Every request is a separate copy of your document with the fact inserted once. "
                           "The fact is shown between >> and <<.")
                for r in doc_rows:
                    st.markdown(f"**{r['model']} · depth {r['depth_percent']}%** "
                                f"(landed about {r.get('actual_position', 0):g}% of the way through)")
                    st.code(r.get("placement", ""), language=None)

# ================================================================ TAB 3: saved results
with tab_saved:
    st.subheader("Saved results")
    st.caption("The big command-line runs, read from results/results.csv (one fact) and "
               "results/multi_results.csv (many facts). Too long to run live, so they are saved as evidence.")

    single_p, multi_p = ROOT / "results" / "results.csv", ROOT / "results" / "multi_results.csv"

    if single_p.exists():
        s, e = split_saved(pd.read_csv(single_p), ["model", "length_tokens", "depth_percent", "trial"])
        s["is_correct"] = s["correct"].astype(str).str.lower() == "true"
        st.markdown("#### One fact")
        pick = st.selectbox("Model", sorted(s["model"].unique()), key="saved_single_model")
        sub, sub_e = s[s["model"] == pick], e[e["model"] == pick]
        k1, k2 = st.columns(2)
        k1.metric("Trials", len(sub))
        k2.metric("Accuracy", f"{sub['is_correct'].mean():.0%}")
        show_heatmap(sub, sub_e, "length_tokens", "depth_percent", "is_correct",
                     f"{pick}: accuracy by length and depth", "depth")
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
        mm, me = split_saved(pd.read_csv(multi_p), ["model", "length_tokens", "n_facts", "trial"])
        st.markdown("---")
        st.markdown("#### Many facts at once")
        if len(mm) or len(me):
            models_here = sorted(set(mm["model"]) | set(me["model"]))
            pick2 = st.selectbox("Model", models_here, key="saved_multi_model")
            sub2, sub2_e = mm[mm["model"] == pick2], me[me["model"] == pick2]
            show_heatmap(sub2, sub2_e, "length_tokens", "n_facts", "recall",
                         f"{pick2}: share of facts found", "facts")
        if len(mm):
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