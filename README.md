# Context Window Retrieval Stress Tester

An evaluation harness that benchmarks how reliably different LLMs retrieve a
specific fact ("needle") buried inside long documents ("haystacks") of
varying length and position -- a "needle in a haystack" test, the standard
method for measuring long-context retrieval reliability.

## Why this matters

Context windows are marketed as 100K, 200K, even 1M+ tokens, but a large
context window doesn't guarantee a model actually *uses* everything inside
it reliably. This project measures where retrieval starts to break down --
by document length, by where the fact sits in the document, and across
different models -- to answer a practical question: **at what point should
you stop stuffing documents into a prompt and switch to retrieval (RAG)
instead?**

## How it works

1. **Build haystacks** -- long documents assembled from unrelated source
   text, at several target lengths (e.g. 1K, 10K, 50K, 100K tokens).
2. **Insert a needle** -- a specific, unambiguous fact (e.g. a made-up
   code or password) at a controlled depth (0%, 25%, 50%, 75%, 100%
   through the document).
3. **Query the model** -- send the full haystack + a question targeting
   the needle, across multiple models.
4. **Grade and log** -- check whether the model's answer contains the
   correct fact, and log length, depth, model, correctness, latency, and
   token usage to `results/results.csv`.
5. **Visualize** -- plot a heatmap per model (length x depth x accuracy),
   plus accuracy/latency-vs-length line charts.

## Setup

```bash
cd context-stress-tester
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the API keys for the models you'll test
```

### Add source text

Drop a handful of unrelated `.txt` files into `data/sources/` -- these get
stitched together to build the haystacks. Good sources: public domain books
(Project Gutenberg), Wikipedia article dumps, or your own essays/articles.
Aim for at least ~150K tokens of raw source material so you can build the
largest haystacks without too much repetition.

Test text sourced from English Wikipedia (CC BY-SA 4.0), fetched with fetch_wikipedia.py.
## Running the experiment

```bash
# 1. Build haystacks of different sizes
python src/haystack.py --lengths 1000 10000 50000 100000

# 2. Run the model sweep (edit MODEL_REGISTRY in src/model_clients.py
#    to add/remove models first)
python src/run_experiment.py \
    --models gpt-4o-mini claude-haiku \
    --lengths 1000 10000 50000 100000 \
    --depths 0 25 50 75 100 \
    --trials 3

# 3. Generate heatmaps and charts
python src/visualize.py
```

Results land in `results/`:
- `results.csv` -- every individual test run
- `heatmaps.png` -- accuracy heatmap per model
- `accuracy_latency.png` -- accuracy and latency vs. context length
- `summary.csv` -- aggregated accuracy/latency per model x length

## Extending it (for a stronger project)

- **Multi-needle**: use `insert_multiple_needles()` in `src/needle.py` to
  hide 2-5 facts and measure how many are retrieved correctly.
- **More models**: add any OpenAI-compatible or Anthropic-compatible
  endpoint to `MODEL_REGISTRY` in `src/model_clients.py`.
- **Cost tracking**: `prompt_tokens`/`completion_tokens` are already logged
  per call -- multiply by each provider's published per-token pricing to
  add a cost column.
- **Write-up**: turn `results/summary.csv` and the heatmaps into a short
  findings report -- e.g. "Model X stays reliable to 50K tokens but drops
  below 70% accuracy past 100K when the fact sits in the middle third."

## Project structure

```
context-stress-tester/
├── data/
│   ├── sources/       # your raw .txt source documents (not provided)
│   └── haystacks/     # generated haystack files
├── results/
│   ├── results.csv
│   ├── summary.csv
│   ├── heatmaps.png
│   └── accuracy_latency.png
├── src/
│   ├── haystack.py       # builds haystacks from source text
│   ├── needle.py         # inserts facts at controlled depths
│   ├── model_clients.py  # unified interface to OpenAI/Anthropic/Groq
│   ├── run_experiment.py # grid runner: model x length x depth
│   └── visualize.py      # heatmaps + summary charts
├── requirements.txt
└── .env.example
```
