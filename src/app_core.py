"""
app_core.py

The logic behind app.py, with no Streamlit code in it so it can be tested
on its own. Builds test cases, runs one request with retries, grades it.
"""

import random
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


def is_retryable(e):
    status = getattr(e, "status_code", None)
    name = type(e).__name__
    return status in (429, 500, 502, 503, 504) or name in {
        "RateLimitError", "InternalServerError", "APIConnectionError", "APITimeoutError",
    }


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
            model_answer=answer.replace("\n", " | ")[:300],
            latency_sec=round(latency, 2),
            prompt_tokens=usage.get("prompt_tokens"),
            auth_error=False,
        )
    except Exception as e:
        row.update(
            found=None,
            recall=None,
            found_flags=None,
            status="ERROR",
            error_message=redact(f"{type(e).__name__}: {e}"[:300], api_key),
            model_answer="",
            latency_sec=None,
            prompt_tokens=None,
            auth_error=is_auth_error(e),
        )
    return row


def run_own_document(model, document, fact, question, expected, depth, api_key, log=print, sleep=time.sleep):
    """The 'your own document' test: hide the user's fact in the user's text."""
    doc = insert_at_depth(document, fact, depth)
    row = {"model": model, "depth_percent": depth}
    try:
        answer, latency, usage = ask_with_retry(model, doc, question, api_key, log=log, sleep=sleep)
        row.update(
            result="FOUND" if expected.lower() in answer.lower() else "NOT FOUND",
            reply=answer.replace("\n", " | ")[:300],
            latency_sec=round(latency, 2),
            prompt_tokens=usage.get("prompt_tokens"),
            auth_error=False,
        )
    except Exception as e:
        row.update(
            result="ERROR",
            reply=redact(f"{type(e).__name__}: {e}"[:300], api_key),
            latency_sec=None,
            prompt_tokens=None,
            auth_error=is_auth_error(e),
        )
    return row