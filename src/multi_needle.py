"""
multi_needle.py

Multi-needle recall test. Instead of hiding ONE fact, hide N facts at evenly
spaced depths and ask the model to report all of them in a single answer.

Single-fact retrieval is saturated for strong models (every cell scores 100%).
Recalling many facts at once is much harder, so this is where accuracy can
start to drop.

Run from the project root:
    python src\\multi_needle.py --models gemini-3.5-flash-lite --lengths 10000 50000 100000 --n-facts 5 10 20 --trials 3

Rows go to results\\multi_results.csv, one row per request. Re-running skips
combinations already done with status OK, so you can stop (Ctrl+C) and resume.
"""

import argparse
import csv
import random
import time
from itertools import product
from pathlib import Path

import pandas as pd
import tiktoken
from openai import APIConnectionError, InternalServerError, RateLimitError
from tqdm import tqdm

from model_clients import query_model

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources"
OUT = ROOT / "results" / "multi_results.csv"

WAITS = [20, 60, 60, 90]  # seconds between retries (same policy as run_experiment.py)

COLUMNS = [
    "model", "length_tokens", "n_facts", "trial",
    "found", "recall", "found_flags",
    "prompt_tokens", "completion_tokens", "latency_sec",
    "status", "error_message", "model_answer",
]

ADJECTIVES = [
    "Amber", "Cobalt", "Dusty", "Ember", "Frost", "Gilded", "Hollow", "Ivory",
    "Jade", "Kestrel", "Lunar", "Mossy", "Nimbus", "Onyx", "Pewter", "Quartz",
    "Russet", "Saffron", "Tundra", "Umber", "Velvet", "Willow", "Yarrow", "Zephyr",
]
NOUNS = [
    "Harbor", "Ridge", "Depot", "Annex", "Bastion", "Cistern", "Dock", "Estate",
    "Forge", "Garrison", "Hangar", "Isle", "Junction", "Keep", "Lodge", "Mill",
    "Outpost", "Pier", "Quarry", "Relay",
]
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

_ENC = tiktoken.get_encoding("cl100k_base")
_CORPUS_TOKENS = None  # loaded once, reused


def make_facts(n, seed):
    """Return n made-up facts as (name, code, sentence). Same seed -> same facts."""
    rng = random.Random(seed)
    names = rng.sample([f"{a} {b}" for a in ADJECTIVES for b in NOUNS], n)
    used = set()
    facts = []
    for name in names:
        while True:
            code = f"{rng.choice(LETTERS)}{rng.choice(LETTERS)}-{rng.randint(1000, 9999)}"
            if code not in used:
                used.add(code)
                break
        facts.append((name, code, f"The access code for {name} station is {code}."))
    return facts


def build_haystack(target_tokens):
    """First target_tokens tokens of the shuffled source articles (fixed order)."""
    global _CORPUS_TOKENS
    if _CORPUS_TOKENS is None:
        files = sorted(SOURCES.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files in {SOURCES}")
        random.Random(42).shuffle(files)
        text = "\n\n".join(f.read_text(encoding="utf-8") for f in files)
        _CORPUS_TOKENS = _ENC.encode(text)
    if len(_CORPUS_TOKENS) < target_tokens:
        raise ValueError(
            f"Corpus has {len(_CORPUS_TOKENS)} tokens, need {target_tokens}"
        )
    return _ENC.decode(_CORPUS_TOKENS[:target_tokens])


def insert_facts(text, sentences):
    """
    Insert sentence i at depth (i + 0.5) / N, snapped back to the previous
    sentence boundary. Positions are computed on the original text, then
    inserted from the end backwards so earlier positions stay valid.
    """
    n = len(sentences)
    positions = []
    for i in range(n):
        idx = int(len(text) * (i + 0.5) / n)
        boundary = text.rfind(". ", 0, idx)
        positions.append(0 if boundary == -1 else boundary + 2)

    for pos, sentence in sorted(zip(positions, sentences), key=lambda x: x[0], reverse=True):
        text = text[:pos] + sentence + " " + text[pos:]
    return text


def ask(model, document, question):
    """query_model with retries on rate limits, server errors and connection errors."""
    for attempt in range(len(WAITS) + 1):
        try:
            return query_model(model, document, question)
        except (RateLimitError, InternalServerError, APIConnectionError) as e:
            if attempt == len(WAITS):
                raise
            wait = WAITS[attempt]
            print(f"\nRetry {attempt + 1}/{len(WAITS)} after {type(e).__name__}: {str(e)[:200]} -- waiting {wait}s")
            time.sleep(wait)


def pause_after(prompt_tokens):
    if prompt_tokens >= 80_000:
        time.sleep(60)
    elif prompt_tokens >= 30_000:
        time.sleep(15)
    else:
        time.sleep(5)


def done_keys():
    if not OUT.exists():
        return set()
    df = pd.read_csv(OUT)
    df = df[df["status"] == "OK"]
    return {
        (r.model, int(r.length_tokens), int(r.n_facts), int(r.trial))
        for r in df.itertuples()
    }


def write_row(row):
    OUT.parent.mkdir(exist_ok=True)
    new_file = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gemini-3.5-flash-lite"])
    p.add_argument("--lengths", nargs="+", type=int, default=[10000, 50000, 100000])
    p.add_argument("--n-facts", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()

    finished = done_keys()
    jobs = [
        j for j in product(args.models, args.lengths, args.n_facts, range(args.trials))
        if j not in finished
    ]
    print(f"{len(jobs)} requests to run ({len(finished)} already done)")

    for model, length, n, trial in tqdm(jobs, desc="Multi-needle"):
        facts = make_facts(n, seed=trial * 100 + n)  # depends on trial and n, not length
        doc = insert_facts(build_haystack(length), [f[2] for f in facts])

        asked = [f[0] for f in facts]
        random.Random(trial).shuffle(asked)  # question order differs from document order
        question = (
            "The document contains records of the form 'The access code for X station is CODE.' "
            "Give the access code for each of these stations: " + "; ".join(asked) + ". "
            "Reply with one line per station in the format 'station: code'. "
            "If a station is not in the document, write 'station: NOT FOUND'."
        )

        row = {"model": model, "length_tokens": length, "n_facts": n, "trial": trial}
        try:
            answer, latency, usage = ask(model, doc, question)
            flags = [1 if code.lower() in answer.lower() else 0 for _, code, _ in facts]
            row.update(
                found=sum(flags), recall=round(sum(flags) / n, 4),
                found_flags=",".join(map(str, flags)),  # ordered by depth: first = shallowest
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                latency_sec=round(latency, 3),
                status="OK", error_message="", model_answer=answer.replace("\n", " | "),
            )
            write_row(row)
            pause_after(usage["prompt_tokens"])
        except Exception as e:
            print(f"\nERROR ({type(e).__name__}): {str(e)[:300]}")
            row.update(status="ERROR", error_message=str(e)[:300])
            write_row(row)
            time.sleep(60)

    if OUT.exists():
        df = pd.read_csv(OUT)
        df = df[df["status"] == "OK"]
        df = df.drop_duplicates(["model", "length_tokens", "n_facts", "trial"], keep="last")
        print("\nMean recall (fraction of facts found) by length x n_facts:")
        print(df.groupby(["model", "length_tokens", "n_facts"])["recall"].mean().unstack())


if __name__ == "__main__":
    main()
