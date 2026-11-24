# DECISIONS.md

A log of the design decisions in the Context Window Retrieval Stress Tester: what I chose, what I considered, and why.

---

## 1. Project choice
- **Decision:** Build #25, a needle-in-a-haystack harness, rather than the first pick (#14 GraphRAG).
- **Why:** I wanted a project I could finish, understand fully and demo. It is narrower than GraphRAG, so I made it stronger with several needle types, long contexts, a write-up and a live demo.
- **Trade-off:** It shows evaluation skill, not model-building. It should be paired with a project that builds something.

## 2. Failed requests are not wrong answers
- **Problem:** The prototype logged API errors as `correct = False`, which made provider failures look like model mistakes.
- **Decision:** Each row has `status` (`OK` or `ERROR`) and `error_message`. On errors, `correct` is left empty. Wrong answers stay as `OK` rows with `correct = False`.
- **Consequence:** `visualize.py` filters to `OK` rows before computing accuracy, and compares `astype(str).str.lower() == "true"` instead of using `astype(bool)`, because an empty value would become `True`.

## 3. Retry policy
- **Retried:** 429 (rate limit), 5xx (server errors) and connection/timeout errors.
- **Not retried:** 400, 401 and 404. These are my mistakes (bad request, bad key, wrong model name), and retrying cannot fix them.
- **Waits:** `[20, 60, 60, 90]` seconds. The earlier, shorter waits failed on 100K prompts. A retry re-sends the whole prompt, so it spends the per-minute token budget again (250K/min), and quick retries drained it.
- **Client setting:** `max_retries=0` on the OpenAI client. By default the library retries silently, which hid failures and doubled token use.

## 4. Pauses between calls
- **Decision:** Pause by prompt size: 5 s (under 30K tokens), 15 s (30K and over), 60 s (80K and over).
- **Why:** A fixed 0.5 s sleep did not prevent rate limits. Large prompts use the per-minute token budget much faster than small ones.

## 5. Provider and model
- **Groq free tier rejected:** the 8K tokens/min cap blocked anything beyond about 7K tokens.
- **Compared:** Gemini (up to 1M context), Mistral (large free limits, but phone verification and a data-training opt-in), Cerebras (64K free-trial context, credit-based), OpenRouter free router (random model, so unusable for a benchmark).
- **Chosen:** Google Gemini through its OpenAI-compatible endpoint, so I could keep the `openai` SDK.
- **Model:** `gemini-3.5-flash-lite`. The `3.6`, `3.7` and `3.8` flash models kept returning 503s or timeouts. Limits on my key: 15 requests/min, 250K input tokens/min, 500 requests/day.
- **Reasoning setting:** `reasoning_effort="minimal"` is rejected with a 400, so I use `"low"`.
- **Limit of this choice:** results describe this one model only. A second model is future work.

## 6. Corpus
- **Problem:** My 8 original articles were only about 3,500 tokens, so long haystacks repeated text, and the model could in principle exploit the repetition.
- **Decision:** Add 14 English Wikipedia articles (fetched with `fetch_wikipedia.py`) to the 8 originals, for about 243K tokens in total.
- **Attribution:** "Test text sourced from English Wikipedia (CC BY-SA 4.0)." This is in the README.
- **Known gap:** 11 articles failed to download (Wikipedia returned 429), and I did not retry them. The corpus is large enough for 100K tests, but not for 200K without repeating text.

## 7. Needle design
- **Decision:** 12 needles in 4 types, 3 each, each tagged with a `needle_type`:
  - **baseline:** a single obvious fact.
  - **blended:** a fact written in the same style as the surrounding text.
  - **distractor:** a similar-looking decoy fact elsewhere in the text.
  - **reworded:** a question phrased differently from the needle.
- **Why:** Single obvious facts in short text are easy for modern models, so an all-baseline test gives 100% everywhere and shows nothing. Accuracy can only drop under harder conditions.
- **Rotation:** `needle = DEFAULT_NEEDLES[trial % len(DEFAULT_NEEDLES)]`, so every cell cycles through all 12 needles.
- **Deferred:** a fifth type, where the answer needs two facts from two places.

## 8. Grading
- **Decision:** Case-insensitive substring match of the expected answer inside the model's answer.
- **Why:** Models often reply with extra words ("Room Zebra-7" for "Zebra-7"), so exact match would mark correct answers wrong.
- **Known weakness:** A model that lists several candidates could pass by accident (false positive). A model-based grader would be stricter. Future work.

## 9. Token counting
- **Decision:** Count tokens with `tiktoken` (`cl100k_base`), not the model's own tokenizer.
- **Check:** Gemini counted my "100K" test as 101,489 tokens, so my estimate is about 1.5% off. Acceptable, and reported honestly.

## 10. Reproducibility
- **Decision:** Seed 42 for haystack construction, temperature 0, depth snapped back to the previous sentence boundary so the needle never splits a sentence.
- **Trials:** Several trials per cell, because a single miss can be noise. Example: one miss at 10K / 25% depth did not reproduce across 6 re-runs, so I treat it as noise, not a pattern. That cell now has extra trials, so its accuracy is not directly comparable with other cells.

## 11. Streamlit app
- **API keys:** Visitors enter their own key. It is kept for the session only and never saved. My key is never in a public app.
- **Safety:** The app reuses the retry and pause logic and caps the number of requests per run.
- **Honest framing:** The app tests whether a model can find *your* fact in *your* document. It does not find facts itself.
- **Look:** A "lab notebook" style (warm paper background, ink text, one rust accent, serif headings, monospace numbers, no gradients or emoji) to avoid the generic purple AI-app look.

## 12. Why results show 100% so far
- **Observation:** Every OK row so far, across 10K and 100K tokens and all four needle types, is correct.
- **Interpretation:** This model handles these tests easily at these lengths. It does not mean the harness is broken, because the harness can record misses (one was recorded earlier, a `NOT FOUND` at 10K), and errors are kept separate from wrong answers.
- **Honest conclusion:** A null result at this range. To find a breaking point I would need longer contexts, harder needles, multi-fact questions or weaker models.

---

## Still to add (fill in when done)
- [ ] Final results for the 50K and 10K runs
- [ ] Accuracy by needle type and length (table)
- [ ] Second model comparison (optional)
- [ ] Lessons learned: the CSV header is written only when the file is created, so rename the old `results.csv` after changing columns


On gemini-3.5-flash-lite, retrieval was perfect (108/108) across context lengths of 10K, 50K and 100K tokens, three insertion depths, and four needle types (baseline, blended, distractor, reworded). No positional effect was observed. With 12 trials per cell, this rules out large failure rates but not small ones (~3% overall). The harness can register failures, as an earlier non-reproducible miss showed. Breaking this model likely requires longer contexts, multi-fact questions or weaker models.