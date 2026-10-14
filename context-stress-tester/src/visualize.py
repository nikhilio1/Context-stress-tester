"""
visualize.py

Reads results/results.csv and produces one heatmap per model:
    x-axis = needle depth (%)
    y-axis = haystack length (tokens)
    color  = accuracy (0-1) across trials in that cell

Also prints a summary table and a latency/cost-vs-length chart.

Usage:
    python src/visualize.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_PATH = "results/results.csv"
OUT_DIR = "results"


def load_results() -> pd.DataFrame:
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"{RESULTS_PATH} not found -- run run_experiment.py first.")
    df = pd.read_csv(RESULTS_PATH)
    df["correct"] = df["correct"].astype(bool)
    return df


def plot_heatmaps(df: pd.DataFrame):
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = df[df["model"] == model]
        pivot = sub.pivot_table(
            index="length_tokens", columns="depth_percent", values="correct", aggfunc="mean"
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0%",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Accuracy"},
        )
        ax.set_title(model)
        ax.set_xlabel("Needle depth (%)")
        ax.set_ylabel("Haystack length (tokens)")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "heatmaps.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close()


def plot_latency_cost(df: pd.DataFrame):
    summary = (
        df.groupby(["model", "length_tokens"])
        .agg(
            accuracy=("correct", "mean"),
            avg_latency=("latency_sec", "mean"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model in summary["model"].unique():
        sub = summary[summary["model"] == model]
        axes[0].plot(sub["length_tokens"], sub["accuracy"], marker="o", label=model)
        axes[1].plot(sub["length_tokens"], sub["avg_latency"], marker="o", label=model)

    axes[0].set_title("Accuracy vs. context length")
    axes[0].set_xlabel("Haystack length (tokens)")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].set_title("Latency vs. context length")
    axes[1].set_xlabel("Haystack length (tokens)")
    axes[1].set_ylabel("Avg. latency (sec)")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "accuracy_latency.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close()

    summary_path = os.path.join(OUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


def main():
    df = load_results()
    plot_heatmaps(df)
    plot_latency_cost(df)
    print("\n--- Summary ---")
    print(df.groupby("model")["correct"].mean().to_string())


if __name__ == "__main__":
    main()
