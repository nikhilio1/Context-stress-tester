import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

RESULTS_PATH = "results/results.csv"

st.set_page_config(page_title="Context Window Stress Tester", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&family=Newsreader:opsz,wght@6..72,500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'Newsreader', serif !important; font-weight: 500; letter-spacing: -0.01em; }
.block-container { max-width: 980px; padding-top: 3rem; }
.statline { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; color: #6B665C;
            border-top: 1px solid #D9D3C5; border-bottom: 1px solid #D9D3C5;
            padding: .6rem 0; margin: 1rem 0 2rem; }
footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
st.title("Context Window Retrieval Stress Tester")
st.caption("Does the model find a hidden fact, by text length and by position?")

if not os.path.exists(RESULTS_PATH):
    st.error(f"{RESULTS_PATH} not found. Run run_experiment.py first.")
    st.stop()

df = pd.read_csv(RESULTS_PATH)
if "status" not in df.columns:
    df["status"] = "OK"
if "needle_type" not in df.columns:
    df["needle_type"] = "baseline"
df["needle_type"] = df["needle_type"].fillna("baseline")

# Failed requests are excluded; only real answers count.
ok = df[df["status"] == "OK"].copy()
ok["correct"] = ok["correct"].astype(str).str.lower() == "true"

acc = f"{ok['correct'].mean():.1%}" if len(ok) else "n/a"
st.markdown(
    f'<div class="statline">{len(df)} requests &nbsp;·&nbsp; '
    f'{len(df) - len(ok)} failed, excluded &nbsp;·&nbsp; {acc} correct</div>',
    unsafe_allow_html=True,
)

if ok.empty:
    st.warning("No successful requests yet.")
    st.stop()

model = st.selectbox("Model", sorted(ok["model"].unique()))
all_types = sorted(ok["needle_type"].unique())
types = st.multiselect("Needle types", all_types, default=all_types)
view = ok[(ok["model"] == model) & (ok["needle_type"].isin(types))]

if view.empty:
    st.warning("Nothing to show for this selection.")
    st.stop()

pivot = view.pivot_table(
    index="length_tokens", columns="depth_percent", values="correct", aggfunc="mean"
)
from matplotlib.colors import LinearSegmentedColormap

PAPER, INK, RUST = "#F4F1EA", "#1F3A4D", "#B5441F"
cmap = LinearSegmentedColormap.from_list("paper", [RUST, PAPER, INK])

fig, ax = plt.subplots(figsize=(7, 3.6))
fig.patch.set_facecolor(PAPER)
ax.set_facecolor(PAPER)
sns.heatmap(pivot, annot=True, fmt=".0%", vmin=0, vmax=1, cmap=cmap,
            linewidths=2, linecolor=PAPER, cbar=False, ax=ax,
            annot_kws={"family": "monospace", "size": 10})
ax.set_xlabel("needle depth, % of text")
ax.set_ylabel("")
ax.set_yticklabels([f"{int(v):,}" for v in pivot.index], rotation=0)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
st.pyplot(fig)

st.subheader("Accuracy by needle type")
by_type = view.groupby("needle_type")["correct"].agg(accuracy="mean", requests="count")
st.dataframe(by_type)

st.subheader("Misses")
misses = view[~view["correct"]]
st.dataframe(
    misses[["length_tokens", "depth_percent", "needle_type", "expected_answer", "model_answer"]]
)