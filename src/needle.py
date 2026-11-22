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
    # --- baseline (your original 3) ---
    {
        "type": "baseline",
        "needle": "The secret launch code for Project Aurora is 48291.",
        "question": "What is the secret launch code for Project Aurora?",
        "answer": "48291",
    },
    {
        "type": "baseline",
        "needle": "The hidden meeting room for the quarterly budget review is Room Zebra-7.",
        "question": "Which room is the quarterly budget review held in?",
        "answer": "Zebra-7",
    },
    {
        "type": "baseline",
        "needle": "The backup server's root password was temporarily set to 'Tangerine-Moose-19'.",
        "question": "What was the backup server's root password temporarily set to?",
        "answer": "Tangerine-Moose-19",
    },
    # --- blended in: reads like the surrounding encyclopedia text ---
    {
        "type": "blended",
        "needle": "The Valmora Accord was negotiated in the harbour town of Quenby-on-Tarn, where the delegates met over three weeks.",
        "question": "In which town was the Valmora Accord negotiated?",
        "answer": "Quenby-on-Tarn",
    },
    {
        "type": "blended",
        "needle": "The Drennick Expedition of 1831 was led by a cartographer named Ostrel Havenmoor, who charted the northern estuaries.",
        "question": "Who led the Drennick Expedition?",
        "answer": "Ostrel Havenmoor",
    },
    {
        "type": "blended",
        "needle": "The Brightwater Canal, completed after eleven years of construction, was financed chiefly by the Marlowe-Dessin Trading Company.",
        "question": "Which company mainly financed the Brightwater Canal?",
        "answer": "Marlowe-Dessin",
    },
    # --- look-alike distractors: three similar facts, one question ---
    {
        "type": "distractor",
        "needle": "The access code for the north vault is 61304. The access code for the east vault is 77215. The access code for the south vault is 38860.",
        "question": "What is the access code for the east vault?",
        "answer": "77215",
    },
    {
        "type": "distractor",
        "needle": "Engineer Pria Kolvane moved to Karvenna in March. Engineer Tomas Ruiz moved to Dunmoor in June. Engineer Wen Adeyemi moved to Pellisford in September.",
        "question": "Which city did Tomas Ruiz move to?",
        "answer": "Dunmoor",
    },
    {
        "type": "distractor",
        "needle": "Server Alpha-3 is reserved for the project named Cobalt-Finch. Server Alpha-5 is reserved for the project named Maple-Quartz. Server Alpha-8 is reserved for the project named Ember-Lotus.",
        "question": "Which project is Server Alpha-5 reserved for?",
        "answer": "Maple-Quartz",
    },
    # --- reworded question: almost no shared words with the fact ---
    {
        "type": "reworded",
        "needle": "The finance team's weekly sync takes place in the Pelican Annex.",
        "question": "Where do the people who handle the budget meet each week?",
        "answer": "Pelican Annex",
    },
    {
        "type": "reworded",
        "needle": "Every Thursday, the night watchman at the Harrow Street depot feeds the stray heron he calls Bartholomew Quill.",
        "question": "What name does the depot's night guard give to the bird he looks after?",
        "answer": "Bartholomew Quill",
    },
    {
        "type": "reworded",
        "needle": "The cafeteria's special soup this season is a smoked-fig bisque, which the staff nicknamed 'Plum Thunder'.",
        "question": "What nickname did employees give the seasonal dish served at lunch?",
        "answer": "Plum Thunder",
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
