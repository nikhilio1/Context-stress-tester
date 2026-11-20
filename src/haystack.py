"""
haystack.py

Builds long "haystack" documents of a target token length by concatenating
source text files. Put any unrelated .txt files (Wikipedia articles, essays,
news articles, etc.) into data/sources/ and this script will stitch them
together, repeating/shuffling as needed, until it hits each target length.

Usage:
    python src/haystack.py --sources data/sources --out data/haystacks \
        --lengths 1000 10000 50000 100000 --model gpt-4o
"""

import argparse
import os
import random
import glob

import tiktoken


def load_source_texts(source_dir: str) -> list[str]:
    """Load all .txt files from the source directory."""
    paths = sorted(glob.glob(os.path.join(source_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(
            f"No .txt files found in {source_dir}. "
            "Add some source documents first (see README)."
        )
    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if content:
                texts.append(content)
    return texts


def build_haystack(
    source_texts: list[str],
    target_tokens: int,
    encoding_name: str = "cl100k_base",
    seed: int = 42,
) -> str:
    """
    Concatenate (and repeat/shuffle if needed) source texts until the
    haystack reaches approximately target_tokens tokens. Trims to the
    exact token count at the end.
    """
    enc = tiktoken.get_encoding(encoding_name)
    rng = random.Random(seed)

    pool = source_texts.copy()
    rng.shuffle(pool)

    chunks = []
    total_tokens = 0
    i = 0
    while total_tokens < target_tokens:
        if i >= len(pool):
            # ran out of unique source docs -- reshuffle and reuse
            rng.shuffle(pool)
            i = 0
        chunk = pool[i]
        chunks.append(chunk)
        total_tokens += len(enc.encode(chunk))
        i += 1

    full_text = "\n\n".join(chunks)
    tokens = enc.encode(full_text)
    trimmed = enc.decode(tokens[:target_tokens])
    return trimmed


def main():
    parser = argparse.ArgumentParser(description="Build haystack documents.")
    parser.add_argument("--sources", default="data/sources")
    parser.add_argument("--out", default="data/haystacks")
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[1000, 10000, 50000, 100000],
        help="Target token lengths to generate.",
    )
    parser.add_argument("--encoding", default="cl100k_base")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    source_texts = load_source_texts(args.sources)
    print(f"Loaded {len(source_texts)} source documents.")

    for length in args.lengths:
        haystack = build_haystack(source_texts, length, args.encoding)
        out_path = os.path.join(args.out, f"haystack_{length}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(haystack)
        print(f"Built {out_path} (~{length} tokens)")


if __name__ == "__main__":
    main()
