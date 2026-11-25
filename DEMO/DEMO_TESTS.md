# Demo tests: exact settings and what to expect

Run these in order. Every setting is written exactly as it appears on screen.
"Expected" means what I believe will happen, not a guarantee. A model can surprise you; if it does, say what you see.

## Before you start

- Files in this folder: `demo_document_short.txt` (about 1,800 tokens) and `demo_document_long.txt` (about 12,900 tokens, an estimate; the app prints the real count).
- `.env` holds `GEMINI_API_KEY` and `GROQ_API_KEY`, and the sidebar box **Use keys from my .env file** is ticked.
- Launch: `python -m streamlit run app.py`
- Groq's free tier allows about 8,000 tokens per minute. That limit is what Tests 1 and 6 use on purpose.

## What the colours mean (say this once at the start)

| Colour | Meaning |
|---|---|
| Green | The model found the fact |
| Red | The model answered and missed it (a wrong answer) |
| Hatched deep red, "FAILED" | The request itself failed (too large, rate limit, bad key). This is an error, not a wrong answer, and it is kept out of the accuracy numbers |

## At a glance: fact, question, expected answer, depth

### Tab 1 tests: you type NO fact, question or answer
The app makes them up. With **Trials per setting = 1** they are always exactly these, every run.

| Test | Fact the app hides | Question the app asks | Expected answer | Depth |
|---|---|---|---|---|
| 1 (at 2,000 and at 10,000 tokens) | `The access code for Willow Junction station is NP-1663.` | `What is the access code for Willow Junction station?` | `NP-1663` | 0% (start) |
| 1 (same two lengths) | `The access code for Ember Lodge station is QQ-3863.` | `What is the access code for Ember Lodge station?` | `QQ-3863` | 100% (end) |
| 2, 5 facts | 5 codes: Quartz Quarry `YZ-9684` (10%), Hollow Hangar `AQ-5080` (30%), Tundra Relay `WB-3569` (50%), Kestrel Annex `DM-8685` (70%), Velvet Estate `HN-9908` (90%) | One question asking for all 5 stations, in shuffled order | All 5 codes | Spread evenly (shown above) |
| 2, 10 facts | 10 codes, one every 10%, starting at 5% | One question asking for all 10 stations | All 10 codes | 5%, 15% ... 95% |

Test 2 is scored by **recall**: the share of the codes found. 5 of 5 is 100%, 4 of 5 is 80%.
Test 2b (bad key) uses the same fact as Test 1 at depth 50.

### Tab 2 tests: your own document

Tab 2 has two modes. **Facts already in my document** is the main one: nothing is hidden or added, the model is asked about facts that really are in the text.

Paste these three lines into the questions box (one per line). Each answer exists exactly once in both demo documents:

```
Who was the library's founding librarian? | Edmund Thackeray
How much did the annual book sale raise? | 7,413
Who donated the oldest item in the rare books room? | Calloway
```

| Test | Mode | Document | Models | Expected |
|---|---|---|---|---|
| 3 | Facts already in my document | short | Gemini | 3 FOUND (answers at about 18%, 52%, 86%) + control ABSTAINED |
| 4 | Facts already in my document | short, with 2 bad lines added | Gemini | 2 skipped before running, nothing wasted |
| 5 | Plant a fact (stress test) | short | Gemini | 3 FOUND, then change the question: 3 NOT FOUND |
| 6 | Facts already in my document | long | Groq only | Every box FAILED (too large) |
| 7 | Facts already in my document | long | Gemini only | 3 FOUND + control ABSTAINED |

FOUND needs the expected answer in the reply and the reply must not say NOT FOUND. The **control question** asks about something invented ("the Zephyrine Quasar-9 turbine"); the right reply is NOT FOUND (shown as ABSTAINED). If a model answers it (MADE UP), its FOUND results are less trustworthy.

### Do the Tab 1 settings carry over to Tab 2? No.

| Setting | Tab 1 | Tab 2 |
|---|---|---|
| Models to test | Sidebar | The same sidebar choice, shared |
| API keys / .env box | Sidebar | The same, shared |
| Trials per setting | Its own slider | **None.** 1 request per model per question (plus 1 control), or per depth in plant mode |
| Text length | Its own list (2,000 ... 100,000) | **None.** The length is whatever document you upload; the app prints "about N tokens" |
| Depths | Its own list | Its own separate list ("Hide it at depth (%)") |
| Request cap (36) | Counted separately | Counted separately (models x (questions + 1), or models x depths) |

**Before each Tab 2 test, check the sidebar.** If both models are ticked, Tab 2 runs both: Test 3 would run 8 requests, and the Groq ones would fail on a long document. Tests 3, 4, 5 and 7 need Gemini only. Test 6 needs `gpt-oss-20b` only.

---

## Tab 1: Run a test

### Test 1: Groq fails at 10,000 tokens (about 1 minute)

| Field | Value |
|---|---|
| Models to test | `gpt-oss-20b` only (remove Gemini) |
| What to hide | One fact |
| Trials per setting | 1 |
| Text length (tokens) | 2,000 and 10,000 |
| Hide the fact at depth (%) | 0 and 100 |

- **Plan line should say:** 4 requests, about 24,000 tokens sent, roughly 1 min. An orange warning says Groq only allows about 5,000 tokens at a time.
- **Expected heatmap:**
  - Row "2,000 tokens": green 100%, n=1, under both "start 0%" and "end 100%".
  - Row "10,000 tokens": two hatched deep-red boxes reading **FAILED, 1 request**.
  - Sentence under it: "Every answered setting scored 100%... 2 requests failed (hatched red); the reason is below."
  - Red box under it: "2 requests at 10,000 tokens: Too large for this provider's free limit: it allows 8,000 tokens per minute and this request needed about 10,300."
- **Strip:** at 10,000 tokens the marker turns deep-red dotted and says "failed".
- **What to say:** "The same model reads 2,000 tokens easily and cannot even accept 10,000 on this free plan. The app shows that as a failure, not as the model being wrong."
- **If it looks different:**
  - If a 2,000-token row shows red 0%, the model really missed. Say so; it is a valid result.
  - If the log shows "retry 1/4" on a 10,000-token request, stop and tell me. The app should not retry these.

### Test 2: Gemini with many facts at once (about 1 minute)

| Field | Value |
|---|---|
| Models to test | `gemini-3.5-flash-lite` only |
| What to hide | Many facts at once |
| Trials per setting | 1 |
| Text length (tokens) | 10,000 |
| How many facts to hide | 5 and 10 |

- **Plan line should say:** 2 requests, about 20,000 tokens, roughly 1 min.
- **Expected:** green boxes, "5 facts" and "10 facts", most likely 100% each. The strip shows 5, then 10, markers turning green one request at a time. If a marker turns red, the model missed that fact; point at where in the text it was.
- **What to say:** "Finding one fact is easy. Here the model has to report ten at once, in a different order from the document."
- **Optional stretch, your best chance of seeing misses:** length 50,000, facts 20, 1 trial (1 request, about 50,000 tokens, about 1.5 minutes). I cannot promise misses. A clean result is still a finding.

### Test 2b: A bad API key (10 seconds)

| Field | Value |
|---|---|
| Models to test | `gemini-3.5-flash-lite` |
| Sidebar | Untick **Use keys from my .env file**, then type `wrong-key` in the Gemini key box |
| What to hide | One fact, length 2,000, depth 50, trials 1 |

- **Expected:** one hatched deep-red FAILED box, with "The provider rejected the API key." under it. The run stops there.
- **What to say:** "A bad password is a failure of the request, not the model failing the test."
- **Afterwards:** tick the .env box again.

---

## Tab 2: Your own document

Upload the file, choose the mode with **What do you want to test?**, then fill the box that appears.

### Test 3: Real facts, short document (Gemini)

| Field | Value |
|---|---|
| Models to test | `gemini-3.5-flash-lite` only |
| Upload | `demo_document_short.txt` (about 1,8xx tokens) |
| What do you want to test? | Facts already in my document |
| Questions box | The three lines above |

- **Before you click:** a check table shows each answer as "in document at 17.7% / 51.8% / 86%". Plan line: 3 questions + 1 control = 4 requests.
- **Expected:** heatmap with three green FOUND boxes ordered by position in the document, and a control line saying ABSTAINED.
- **What to say:** "The answers were already in the document. The app checked that before running, so FOUND can only mean the model read the text."

### Test 4: The app refuses to fake a result

Add these two lines below the three from Test 3:

```
Who is the CEO? | Zorg
What is Edmund Thackeray's role? | Edmund Thackeray
```

- **Expected:** both are shown as "skipped" (the first answer is not in the document; the second answer is already inside the question). Only 3 questions + control run.
- **What to say:** "If your expected answer is not in the document, the app will not run it. Otherwise a miss would be your typo, not the model's fault."

### Test 5: Plant a fact (the stress test)

| Field | Value |
|---|---|
| Mode | Plant a fact (stress test) |
| Upload | `demo_document_short.txt` |
| Fact to hide | `The maintenance key for Hangar 9 is OSPREY-42.` |
| Question to ask | `What is the maintenance key for Hangar 9?` |
| Expected answer | `OSPREY-42` |
| Hide it at depth (%) | 0, 50, 100 |

- **Expected:** three green FOUND. The fact is not in your file; it is inserted in memory, once per request, and the file on disk is never changed.
- **Then change the question to** `What is the vault code?` **and run again:** three red NOT FOUND, because the document has no vault code. This shows the app does not mark everything as found.

### Test 6: Groq fails on a long document

| Field | Value |
|---|---|
| Models to test | `gpt-oss-20b` only |
| Upload | `demo_document_long.txt` (about 12,900 tokens, orange warning) |
| Mode / questions | Facts already in my document, the three lines |

- **Expected:** 4 hatched deep-red FAILED boxes ("Too large for this provider's free limit: it allows 8,000 tokens per minute and this request needed about 13,000"). No retries.
- **What to say:** "FAILED is not NOT FOUND. The model never got to read the document."

### Test 7: The same long document on Gemini (the contrast)

| Field | Value |
|---|---|
| Models to test | `gemini-3.5-flash-lite` only |
| Upload / questions | Same as Test 6 |

- **Expected:** three green FOUND (answers at about 15%, 50%, 85%) and ABSTAINED on the control. 4 requests of about 13,000 tokens, with short pauses.
- **What to say:** "Same document, same questions. Groq's free plan cannot take it; Gemini reads it and finds all three."

---

## Tab 3: Saved results (no run needed)

- Show the heatmap for the full command-line experiment: 108 trials from 10,000 to 100,000 tokens.
- Show the "Many facts at once" section once you have run `multi_needle.py`.
- **What to say:** "These are the big runs, too long to repeat live. The tool lets anyone repeat a small version on their own model."

## Suggested order and timing

1. Test 3 (warm-up, 1 minute)
2. Test 1 (the failure, 1 minute)
3. Test 7, then Test 6 (the contrast, 2 minutes)
4. Test 2 (many facts, 1 minute)
5. Tab 3 (the evidence)
6. Test 4 or the second half of Test 5 if someone asks "does it ever say not found?"

Take screenshots of a good run beforehand in case the internet fails.