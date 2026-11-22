"""
model_clients.py

Thin, consistent wrappers around different LLM providers so the rest of the
codebase can call any of them the same way:

    answer, latency_sec, usage = query_model("gpt-4o", system, user_prompt)

Add more providers by writing a new _query_xxx function and registering it
in MODEL_REGISTRY.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "information in the provided document. Be concise -- reply with just the "
    "fact asked for, not a full sentence, if possible. If the document does "
    "not contain the answer, say exactly: NOT FOUND."
)


def _query_openai(model: str, document: str, question: str) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user_prompt = f"<document>\n{document}\n</document>\n\nQuestion: {question}"

    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    latency = time.time() - start

    answer = resp.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
    return answer, latency, usage


def _query_anthropic(model: str, document: str, question: str) -> tuple[str, float, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_prompt = f"<document>\n{document}\n</document>\n\nQuestion: {question}"

    start = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0,
    )
    latency = time.time() - start

    answer = resp.content[0].text.strip()
    usage = {
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
    }
    return answer, latency, usage


def _query_gemini(model: str, document: str, question: str) -> tuple[str, float, dict]:
    # Gemini free tier, via its OpenAI-compatible endpoint.
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=60,
        max_retries=0,
    )
    user_prompt = f"<document>\n{document}\n</document>\n\nQuestion: {question}"

    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        reasoning_effort="low",  # "minimal" is too weak, "high" is too slow
    )
    latency = time.time() - start

    answer = resp.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
    return answer, latency, usage


# Maps a short model label (what you'll see in your results CSV/plots) to
# (provider function, actual API model name).
#
# NOTE: model name strings change often -- check each provider's docs
# (platform.openai.com, docs.claude.com, console.groq.com) for the current
# model IDs before you run this, and swap them in below.
MODEL_REGISTRY = {
    "gpt-4o": (_query_openai, "gpt-4o"),
    "gpt-4o-mini": (_query_openai, "gpt-4o-mini"),
    "claude-sonnet": (_query_anthropic, "claude-sonnet-5"),
    "claude-haiku": (_query_anthropic, "claude-haiku-5-5"),
    "gemini-3.5-flash-lite": (_query_gemini, "gemini-3.5-flash-lite"),
}


def query_model(model_label: str, document: str, question: str) -> tuple[str, float, dict]:
    """
    Query a registered model by its short label. Returns (answer, latency_seconds, usage_dict).
    """
    if model_label not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_label}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    fn, api_model_name = MODEL_REGISTRY[model_label]
    return fn(api_model_name, document, question)
