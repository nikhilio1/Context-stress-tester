"""
run_experiment.py

The main experiment runner. For every combination of:
    model x haystack length x needle depth
it builds the haystack+needle, queries the model, grades the answer, and
appends one row to results/results.csv.

Usage:
    python src/run_experiment.py \
        --models gpt-4o-mini claude-haiku \
        --lengths 1000 10000 50000 \
        --depths 0 25 50 75 100 \
        --trials 1
"""

import argparse
import csv
import os
import time

from tqdm import tqdm

from haystack import build_haystack, load_source_texts
from needle import DEFAULT_NEEDLES, insert_needle
from model_clients import query_model

RESULTS_PATH = "results/results.csv"
FIELDNAMES = [
    "model",
    "length_tokens",
    "depth_percent",
    "trial",
    "needle_question",
    "expected_answer",
    "model_answer",
    "correct",
    "latency_sec",
    "prompt_tokens",
    "completion_tokens",
]


def grade(model_answer: str, expected_answer: str) -> bool:
    """Simple substring match, case-insensitive. Good enough for short,
    distinctive facts like codes/passwords/room numbers."""
    return expected_answer.lower() in model_answer.lower()


def append_result(row: dict):
    file_exists = os.path.exists(RESULTS_PATH)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="Model labels from model_clients.MODEL_REGISTRY")
    parser.add_argument("--lengths", nargs="+", type=int, default=[1000, 10000, 50000])
    parser.add_argument("--depths", nargs="+", type=float, default=[0, 25, 50, 75, 100])
    parser.add_argument("--trials", type=int, default=1, help="Repeats per cell, for reliability")
    parser.add_argument("--sources", default="data/sources")
    args = parser.parse_args()

    source_texts = load_source_texts(args.sources)

    combos = [
        (model, length, depth, trial)
        for model in args.models
        for length in args.lengths
        for depth in args.depths
        for trial in range(args.trials)
    ]

    for model, length, depth, trial in tqdm(combos, desc="Running experiment"):
        needle_info = DEFAULT_NEEDLES[trial % len(DEFAULT_NEEDLES)]
        haystack = build_haystack(source_texts, length)
        full_text, _ = insert_needle(haystack, needle_info["needle"], depth)

        try:
            answer, latency, usage = query_model(model, full_text, needle_info["question"])
            correct = grade(answer, needle_info["answer"])
        except Exception as e:
            answer = f"ERROR: {e}"
            latency = 0.0
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            correct = False

        append_result({
            "model": model,
            "length_tokens": length,
            "depth_percent": depth,
            "trial": trial,
            "needle_question": needle_info["question"],
            "expected_answer": needle_info["answer"],
            "model_answer": answer,
            "correct": correct,
            "latency_sec": round(latency, 3),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        })

        time.sleep(0.5)  # be polite to rate limits

    print(f"\nDone. Results appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
