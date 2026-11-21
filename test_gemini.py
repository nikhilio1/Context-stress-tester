import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=30,
    max_retries=0,
)
start = time.time()
try:
    r = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        reasoning_effort="low",
    )
    print("OK:", r.choices[0].message.content)
except Exception as e:
    print("ERROR:", type(e).__name__, e)
print("seconds:", round(time.time() - start, 1))