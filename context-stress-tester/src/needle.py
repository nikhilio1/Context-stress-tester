"""
needle.py

Inserts one or more "needle" facts into a haystack document at controlled
depth positions (0% = very start, 100% = very end).

Usage (as a library):
    from needle import insert_needle, DEFAULT_NEEDLES
    text, meta = insert_needle(haystack_text, needle="...", depth_percent=50)
"""

import tiktoken


# A pool of ready-made needle/question/answer triples you can use out of the
# box. Feel free to write your own -- the key property is that the fact is
# specific, unambiguous, and NOT something the model could guess without
# actually reading it.
DEFAULT_NEEDLES = [
    {
        "needle": "The secret launch code for Project Aurora is 48291.",
        "question": "What is the secret launch code for Project Aurora?",
        "answer": "48291",
    },
    {
        "needle": "The hidden meeting room for the quarterly budget review is Room Zebra-7.",
        "question": "Which room is the quarterly budget review held in?",
        "answer": "Zebra-7",
    },
    {
        "needle": "The backup server's root password was temporarily set to 'Tangerine-Moose-19'.",
        "question": "What was the backup server's root password temporarily set to?",
        "answer": "Tangerine-Moose-19",
    },
]


def insert_needle(
    haystack_text: str,
    needle: str,
    depth_percent: float,
    encoding_name: str = "cl100k_base",
) -> tuple[str, dict]:
    """
    Insert `needle` into `haystack_text` at `depth_percent` (0-100) of the
    way through, snapped to the nearest sentence boundary (a period+space)
    so the needle doesn't land mid-word.

    Returns (new_text, metadata_dict).
    """
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(haystack_text)
    total = len(tokens)

    insert_at_token = int(total * (depth_percent / 100.0))
    insert_at_token = max(0, min(insert_at_token, total))

    before = enc.decode(tokens[:insert_at_token])
    after = enc.decode(tokens[insert_at_token:])

    # Snap to the nearest sentence boundary at or before the depth point, so
    # the needle reads as its own clean sentence rather than splitting one
    # mid-word. `prefix` is everything that stays before the needle;
    # `remainder` is the leftover fragment of the cut sentence (if any),
    # which continues immediately after the needle.
    boundary = before.rfind(". ")
    if boundary != -1:
        prefix = before[: boundary + 2]
        remainder = before[boundary + 2 :]
    else:
        # No sentence boundary found before the depth point -- insert at the
        # raw cut point with no snapping, don't drag `before` after the needle.
        prefix = before
        remainder = ""

    # Guard against gluing the needle directly onto the previous word when
    # `prefix` doesn't already end in whitespace (can happen on the raw-cut
    # fallback above, or right at a token boundary mid-word).
    if prefix and not prefix[-1].isspace():
        prefix += " "

    new_text = f"{prefix}{needle} {remainder}{after}"

    meta = {
        "depth_percent": depth_percent,
        "total_tokens_before_insert": total,
        "needle": needle,
    }
    return new_text, meta


def insert_multiple_needles(
    haystack_text: str,
    needles: list[str],
    depth_percents: list[float],
    encoding_name: str = "cl100k_base",
) -> tuple[str, dict]:
    """
    Insert several needles at several depths in one pass. Depths are
    processed back-to-front so earlier insertions don't shift the token
    offsets for later ones.
    """
    assert len(needles) == len(depth_percents), "needles and depths must match"
    pairs = sorted(zip(needles, depth_percents), key=lambda p: -p[1])

    text = haystack_text
    for needle, depth in pairs:
        text, _ = insert_needle(text, needle, depth, encoding_name)

    return text, {"needles": needles, "depth_percents": depth_percents}
