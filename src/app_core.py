"""
app_core.py

The logic behind app.py, with no Streamlit code in it so it can be tested
on its own. Builds test cases, runs one request with retries, grades it.
"""

import random
import re
import time
from itertools import product

import multi_needle
from model_clients import MODEL_REGISTRY, query_model
from multi_needle import build_haystack, insert_facts, make_facts

WAITS = [20, 60, 60, 90]          # seconds between retries
MAX_REQUESTS = 36                 # hard cap per run, protects visitors' quotas
LENGTH_OPTIONS = [2000, 5000, 10000, 25000, 50000, 100000]

# provider function name -> (display name, .env variable)
PROVIDERS = {
    "_query_gemini": ("Google Gemini", "GEMINI_API_KEY"),
    "_query_groq": ("Groq", "GROQ_API_KEY"),
    "_query_openai": ("OpenAI", "OPENAI_API_KEY"),
    "_query_anthropic": ("Anthropic", "ANTHROPIC_API_KEY"),
}

# Free tiers that cannot take long prompts (tokens per request that are safe)
SAFE_TOKEN_LIMIT = {"Groq": 5000}
# Hard per-minute token allowance. One request bigger than this can never succeed, whatever the wait.
HARD_TPM = {"Groq": 8000}


class TooLarge(Exception):
    """Raised by us when a request is bigger than the provider's whole per-minute allowance."""
    status_code = 413


def provider_of(model_label):
    fn, _ = MODEL_REGISTRY[model_label]
    return PROVIDERS.get(fn.__name__, (fn.__name__, ""))[0]


def corpus_tokens():
    """How many tokens of source text are available for building haystacks."""
    build_haystack(1)  # makes multi_needle load the corpus
    return len(multi_needle._CORPUS_TOKENS)


def insert_at_depth(text, sentence, depth_percent):
    """Insert one sentence at a depth (0 = start, 100 = end), on a sentence boundary."""
    if depth_percent >= 100:
        return text.rstrip() + " " + sentence
    idx = int(len(text) * depth_percent / 100)
    boundary = text.rfind(". ", 0, idx)
    pos = 0 if boundary == -1 else boundary + 2
    return text[:pos] + sentence + " " + text[pos:]


def pause_seconds(prompt_tokens):
    if prompt_tokens >= 80_000:
        return 60
    if prompt_tokens >= 30_000:
        return 15
    return 5


_LIMIT_RE = re.compile(r"Limit\s+(\d[\d,]*).{0,60}?Requested\s+(\d[\d,]*)", re.S | re.I)
_CODE_RE = re.compile(r"error code:\s*(\d{3})", re.I)


def _limit_requested(text):
    """Pull 'Limit 8000 ... Requested 10300' out of a provider message, if present."""
    m = _LIMIT_RE.search(text or "")
    if not m:
        return None
    return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))


def is_too_large(e):
    """A request bigger than the provider's own limit can never succeed, so retrying is pointless."""
    text = str(e)
    if getattr(e, "status_code", None) == 413:
        return True
    lr = _limit_requested(text)
    if lr and lr[1] > lr[0]:
        return True
    return "request too large" in text.lower()


def is_retryable(e):
    if is_too_large(e):
        return False
    status = getattr(e, "status_code", None)
    name = type(e).__name__
    return status in (429, 500, 502, 503, 504) or name in {
        "RateLimitError", "InternalServerError", "APIConnectionError", "APITimeoutError",
    }


def describe(e):
    """Exception -> one string, always carrying the HTTP status code when the exception has one."""
    text = f"{type(e).__name__}: {e}"
    code = getattr(e, "status_code", None)
    if code and not _CODE_RE.search(text):
        text = f"{type(e).__name__}: Error code: {code} - {e}"
    return text


def explain_error(text):
    """Turn a raw provider error into one plain-English sentence for the screen."""
    t = text or ""
    low = t.lower()
    lr = _limit_requested(t)
    m = _CODE_RE.search(t)
    code = int(m.group(1)) if m else None

    if code == 413 or "request too large" in low or (lr and lr[1] > lr[0]):
        if lr:
            return (f"Too large for this provider's free limit: it allows {lr[0]:,} tokens per minute "
                    f"and this request needed {lr[1]:,}. Use a shorter text or a provider with a bigger limit.")
        return "Too large for this provider's limit. Use a shorter text or a provider with a bigger limit."
    if code in (401, 403) or "authentication" in low or "invalid api key" in low or "permission" in low:
        return "The provider rejected the API key."
    if code == 429 or "ratelimit" in low or "rate limit" in low or "quota" in low:
        return "Rate limit: the provider asked us to slow down and the retries ran out. Wait a minute, or run fewer requests."
    if code == 404 or "notfound" in low or "model not found" in low:
        return "The provider does not recognise this model name."
    if code == 400 or "badrequest" in low:
        return "The provider rejected the request settings."
    if (code and code >= 500) or any(x in low for x in ("internalserver", "overloaded", "high demand", "unavailable")):
        return "The provider was overloaded or down for a moment. Try again."
    if any(x in low for x in ("timeout", "timed out", "connection")):
        return "Network problem, or the provider took too long to answer."
    return "Unexpected error: " + t[:120]


def is_auth_error(e):
    return getattr(e, "status_code", None) in (401, 403) or type(e).__name__ == "AuthenticationError"


def redact(text, key):
    """Never let a visitor's key appear in a log, table or CSV."""
    if key:
        text = text.replace(key, "***")
    return text


def ask_with_retry(model, document, question, api_key, log=print, sleep=time.sleep):
    for attempt in range(len(WAITS) + 1):
        try:
            return query_model(model, document, question, api_key)
        except Exception as e:
            limit = HARD_TPM.get(provider_of(model))
            if limit and (getattr(e, "status_code", None) == 429 or type(e).__name__ == "RateLimitError"):
                needed = len(multi_needle._ENC.encode(document)) + 150   # + system prompt and question
                if needed > limit:
                    raise TooLarge(f"Error code: 413 - Request too large: Limit {limit}, Requested {needed}") from e
            if not is_retryable(e) or attempt == len(WAITS):
                raise
            wait = WAITS[attempt]
            log(f"retry {attempt + 1}/{len(WAITS)} after {type(e).__name__}, waiting {wait}s")
            sleep(wait)


def make_jobs(test, models, lengths, params, trials):
    """test is 'single' (params = depths) or 'multi' (params = fact counts)."""
    return [
        {"test": test, "model": m, "length": l, "param": p, "trial": t}
        for m, l, p, t in product(models, lengths, params, range(trials))
    ]


def estimate(jobs):
    n_req = len(jobs)
    tokens = sum(j["length"] for j in jobs)
    seconds = sum(pause_seconds(j["length"]) + 5 for j in jobs)
    return n_req, tokens, seconds


def build_case(job, haystack):
    """Return (document, question, expected_codes) for one job."""
    if job["test"] == "single":
        depth = job["param"]
        name, code, sentence = make_facts(1, seed=job["trial"] * 1000 + int(depth))[0]
        doc = insert_at_depth(haystack, sentence, depth)
        return doc, f"What is the access code for {name} station?", [code]

    n = job["param"]
    facts = make_facts(n, seed=job["trial"] * 100 + n)
    doc = insert_facts(haystack, [f[2] for f in facts])
    asked = [f[0] for f in facts]
    random.Random(job["trial"]).shuffle(asked)
    question = (
        "The document contains records of the form 'The access code for X station is CODE.' "
        "Give the access code for each of these stations: " + "; ".join(asked) + ". "
        "Reply with one line per station in the format 'station: code'. "
        "If a station is not in the document, write 'station: NOT FOUND'."
    )
    return doc, question, [f[1] for f in facts]


def run_job(job, haystack, api_key, log=print, sleep=time.sleep):
    """Run one request. Returns a row dict; failures become status='ERROR', never a wrong answer."""
    doc, question, codes = build_case(job, haystack)
    row = {
        "test": job["test"],
        "model": job["model"],
        "length_tokens": job["length"],
        "depth_percent": job["param"] if job["test"] == "single" else None,
        "n_facts": 1 if job["test"] == "single" else job["param"],
        "trial": job["trial"],
    }
    try:
        answer, latency, usage = ask_with_retry(
            job["model"], doc, question, api_key, log=log, sleep=sleep
        )
        flags = [c.lower() in answer.lower() for c in codes]
        row.update(
            found=sum(flags),
            recall=sum(flags) / len(codes),
            found_flags=",".join("1" if f else "0" for f in flags),  # ordered by depth
            status="OK",
            error_message="",
            error_summary="",
            model_answer=answer.replace("\n", " | ")[:300],
            latency_sec=round(latency, 2),
            prompt_tokens=usage.get("prompt_tokens"),
            auth_error=False,
        )
    except Exception as e:
        full = redact(describe(e), api_key)
        row.update(
            found=None,
            recall=None,
            found_flags=None,
            status="ERROR",
            error_message=full[:500],
            error_summary=explain_error(full),
            model_answer="",
            latency_sec=None,
            prompt_tokens=None,
            auth_error=is_auth_error(e),
        )
    return row


def placement_snippet(doc, fact, width=90):
    """The text around the hidden fact, with the fact marked, so a person can see exactly where it went."""
    i = doc.find(fact)
    if i < 0:
        return ""
    before = doc[max(0, i - width):i].replace("\n", " ")
    after = doc[i + len(fact):i + len(fact) + width].replace("\n", " ")
    return f"{'...' if i > width else ''}{before}>> {fact} <<{after}{'...' if i + len(fact) + width < len(doc) else ''}"


def run_own_document(model, document, fact, question, expected, depth, api_key, log=print, sleep=time.sleep):
    """The 'your own document' test: hide the user's fact in the user's text, ONCE, at ONE depth."""
    doc = insert_at_depth(document, fact, depth)
    at = doc.find(fact)
    row = {"model": model, "depth_percent": depth,
           "actual_position": round(100 * at / max(len(doc), 1), 1) if at >= 0 else None,
           "placement": placement_snippet(doc, fact)}
    try:
        answer, latency, usage = ask_with_retry(model, doc, question, api_key, log=log, sleep=sleep)
        row.update(
            result="FOUND" if grade(answer, expected) else "NOT FOUND",
            reply=answer.replace("\n", " | ")[:300],
            error_summary="",
            latency_sec=round(latency, 2),
            prompt_tokens=usage.get("prompt_tokens"),
            auth_error=False,
        )
    except Exception as e:
        full = redact(describe(e), api_key)
        row.update(
            result="ERROR",
            reply=full[:500],
            error_summary=explain_error(full),
            latency_sec=None,
            prompt_tokens=None,
            auth_error=is_auth_error(e),
        )
    return row


# ====================================================================== facts ALREADY in the user's document
# The user's document is sent exactly as it is. Nothing is inserted. For the result to mean anything:
#   1. every expected answer must really appear in the document (checked here before any request is made),
#   2. the expected answer must not be sitting inside the question itself,
#   3. a control question the document cannot answer is asked too, and the model must say NOT FOUND.

_CONTROLS = [
    ("What is the registered serial number of the Zephyrine Quasar-9 turbine?", ("zephyrine", "quasar")),
    ("What is the maiden name of the harbourmaster's third cousin, Ottoline Quillfeather?", ("ottoline", "quillfeather")),
]


def says_not_found(reply):
    """True when the model declined to answer, using the exact phrase the system prompt asks for."""
    r = (reply or "").strip().strip("\"'`*. ").lower()
    return r.startswith("not found") or "NOT FOUND" in (reply or "")


def grade(reply, expected):
    """FOUND only if the model did not decline AND the expected answer is in its reply (ignoring capitals)."""
    if says_not_found(reply):
        return False
    return expected.strip().lower() in (reply or "").lower()


def parse_qa(text):
    """One 'question | expected answer' per line -> (items, problems)."""
    items, problems = [], []
    for n, line in enumerate((text or "").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            problems.append({"line": n, "text": line, "why": "missing the | between the question and the expected answer"})
            continue
        q, a = (p.strip() for p in line.rsplit("|", 1))
        if not q or not a:
            problems.append({"line": n, "text": line, "why": "the question or the expected answer is empty"})
            continue
        items.append({"line": n, "question": q, "expected": a})
    return items, problems


def locate(doc, expected):
    """(percent of the way through the document where the expected answer first appears, how many times) or (None, 0)."""
    low, exp = doc.lower(), expected.strip().lower()
    i = low.find(exp)
    if i < 0:
        return None, 0
    return round(100 * i / max(len(doc), 1), 1), low.count(exp)


def check_items(doc, items):
    """Split items into (runnable, skipped). Runnable ones carry the position where the answer sits."""
    runnable, skipped = [], []
    for k, it in enumerate(items, 1):
        pos, count = locate(doc, it["expected"])
        if it["expected"].strip().lower() in it["question"].lower():
            skipped.append({**it, "why": "the expected answer appears inside the question itself, so FOUND would mean nothing"})
        elif pos is None:
            skipped.append({**it, "why": "this answer is not in your document (check the spelling), and a model cannot find what is not there"})
        else:
            runnable.append({**it, "qnum": k, "position": pos, "occurrences": count})
    return runnable, skipped


def control_question(doc):
    """A question that no ordinary document can answer. Picks one whose key words are not in this document."""
    low = doc.lower()
    for q, words in _CONTROLS:
        if not any(w in low for w in words):
            return q
    return _CONTROLS[0][0]


def _row_error(row, e, api_key):
    full = redact(describe(e), api_key)
    row.update(result="ERROR", reply=full[:500], error_summary=explain_error(full), latency_sec=None,
               prompt_tokens=None, auth_error=is_auth_error(e))


def run_question(model, document, item, api_key, log=print, sleep=time.sleep):
    """Ask about a fact that is already in the document. The document is sent unchanged."""
    row = {"kind": "question", "model": model, "qnum": item["qnum"], "question": item["question"],
           "expected": item["expected"], "position": item["position"], "occurrences": item["occurrences"]}
    try:
        answer, latency, usage = ask_with_retry(model, document, item["question"], api_key, log=log, sleep=sleep)
        row.update(result="FOUND" if grade(answer, item["expected"]) else "NOT FOUND",
                   reply=answer.replace("\n", " | ")[:300], error_summary="", latency_sec=round(latency, 2),
                   prompt_tokens=usage.get("prompt_tokens"), auth_error=False)
    except Exception as e:
        _row_error(row, e, api_key)
    return row


def run_control(model, document, api_key, log=print, sleep=time.sleep):
    """Ask something the document cannot answer. A trustworthy model must say NOT FOUND."""
    q = control_question(document)
    row = {"kind": "control", "model": model, "question": q}
    try:
        answer, latency, usage = ask_with_retry(model, document, q, api_key, log=log, sleep=sleep)
        row.update(result="ABSTAINED" if says_not_found(answer) else "MADE UP",
                   reply=answer.replace("\n", " | ")[:300], error_summary="", latency_sec=round(latency, 2),
                   prompt_tokens=usage.get("prompt_tokens"), auth_error=False)
    except Exception as e:
        _row_error(row, e, api_key)
    return row