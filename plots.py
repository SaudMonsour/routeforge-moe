"""Render measured learning, routing and CPU inference behavior."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from prepare import ROOT


def main():
    metrics = json.loads((ROOT/"runs/metrics.json").read_text())
    benchmark = json.loads((ROOT/"runs/benchmark.json").read_text())
    routing = json.loads((ROOT/"runs/routing_diagnostics.json").read_text())
    with (ROOT/"runs/training_history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    output = ROOT/"figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150,
                         "savefig.dpi": 170, "axes.spines.top": False, "axes.spines.right": False})
    roles = ("dense", "unbalanced", "balanced")
    names = {"dense": "Dense", "unbalanced": "MoE · no balancing", "balanced": "MoE · balancing"}
    colors = {"dense": "#667788", "unbalanced": "#c58029", "balanced": "#285aa0"}
    def save(fig, name):
        fig.savefig(output/f"{name}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for role in roles:
        rows = [r for r in history if r["model"] == role]
        ax.plot([int(r["step"]) for r in rows], [float(r["validation_nll"]) for r in rows], marker="o", color=colors[role], label=names[role])
    ax.set(xlabel="Optimizer step", ylabel="Validation NLL (nats / byte)", title="Same byte windows and training budget")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "training-curves")
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    values = [metrics["unigram"]["test_byte_perplexity"], *[metrics["models"][r]["test"]["byte_perplexity"] for r in roles]]
    bars = ax.bar(["Unigram", "Dense", "MoE\nunbalanced", "MoE\nbalanced"], values, color=["#a0a8b0", *[colors[r] for r in roles]])
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4)
    ax.set(ylim=(0, max(values)*1.18), ylabel="Byte perplexity (lower is better)", title="Contiguous holdout · 42,112 next-byte targets")
    save(fig, "model-comparison")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained", sharey=True)
    for layer, ax in enumerate(axes):
        for shift, role in ((-.18, "unbalanced"), (.18, "balanced")):
            fractions = metrics["models"][role]["test"]["routing"]["fractions"][layer]
            ax.bar(np.arange(4)+shift, fractions, width=.36, color=colors[role], label=names[role])
        ax.axhline(.25, linestyle="--", color="#888888", linewidth=1, label="Uniform reference")
        ax.set(xticks=np.arange(4), xlabel="Expert ID", title=f"Decoder layer {layer+1}", ylim=(0, .9))
    axes[0].set_ylabel("Fraction of input tokens routed to expert")
    axes[1].legend(fontsize=9)
    fig.suptitle("Balancing loss reduced concentrated expert usage")
    save(fig, "expert-load")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, batch in zip(axes, (1, 16)):
        rows = [r for r in benchmark["summaries"] if r["batch"] == batch]
        medians = [r["median_ms"] for r in rows]
        bars = ax.bar(["Dense", "MoE\nunbalanced", "MoE\nbalanced"], medians, color=[colors[r] for r in roles])
        ax.errorbar(np.arange(3), medians, yerr=[np.zeros(3), [r["p95_ms"]-r["median_ms"] for r in rows]], fmt="none", color="#23313f", capsize=4)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in medians], padding=3, zorder=5,
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        ax.set(ylabel="Full forward-pass latency (ms)", title=f"Batch {batch}", ylim=(0, max(r["p95_ms"] for r in rows)*1.2))
    fig.suptitle("64-byte sequences · CPU float32\nBars: median · whiskers: p95 · 120 observations per variant")
    save(fig, "latency")
    traces = [routing[r]["trace"]["tokens"] for r in ("unbalanced", "balanced")]
    matrix = np.array([[t["expert_ids"][layer] for t in trace] for trace in traces for layer in range(2)])
    fig, ax = plt.subplots(figsize=(12, 3.8), layout="constrained")
    cmap = ListedColormap(["#456990", "#edae49", "#4a9d82", "#a464a0"])
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=BoundaryNorm(np.arange(-.5, 4.5), 4))
    chars = [chr(t["byte_id"]) if t["byte_id"] != 32 else "·" for t in traces[0]]
    ax.set(xticks=np.arange(len(chars)), xticklabels=chars, yticks=np.arange(4),
           yticklabels=["Unbalanced L1", "Unbalanced L2", "Balanced L1", "Balanced L2"],
           xlabel="Input byte position (· denotes a space)", title="Actual top-one choices for: The creature opened his eyes.")
    ax.tick_params(axis="x", labelsize=9)
    fig.colorbar(image, ax=ax, ticks=range(4), label="Expert ID")
    save(fig, "route-trace")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for row, role in enumerate(("unbalanced", "balanced")):
        for layer in range(2):
            ax = axes[row, layer]
            counts = np.asarray(routing[role]["counts_by_layer_category_expert"])[layer]
            totals = counts.sum(-1)
            normalized = np.divide(counts, totals[:, None], out=np.zeros_like(counts, dtype=float), where=totals[:, None] > 0)
            image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            labels = [f"{name} (n={n:,})" for name, n in zip(routing[role]["categories"], totals)]
            ax.set(xticks=range(4), yticks=range(5), yticklabels=labels, xlabel="Expert ID", title=f"{names[role]} · layer {layer+1}")
            for y in range(5):
                for x in range(4):
                    ax.text(x, y, f"{normalized[y, x]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if normalized[y, x] > .55 else "#142539")
    fig.colorbar(image, ax=axes, label="Fraction within input-byte category", shrink=.8)
    fig.suptitle("Descriptive routing patterns; categories do not establish semantic specialization")
    save(fig, "category-routing")


if __name__ == "__main__":
    main()
