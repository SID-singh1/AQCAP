"""
Paper figures from real run outputs (results/*.jsonl, results/sensitivity_*.csv, paper/summary.csv).

Writes paper/figures/{pareto,lowbit_selection,sensitivity_depth,score_shift}.pdf

Usage:
    python scripts/summarize_results.py --results_dir results --out paper/summary.csv
    python scripts/make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "paper/figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29c"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def load_scores(run_id):
    rows = [json.loads(l) for l in open(f"results/{run_id}.jsonl", encoding="utf-8")]
    return pd.DataFrame([r for r in rows if r.get("type") == "sample"]).sort_values("idx")


def pareto(summary):
    d = summary[(summary.model == "qwen7b") & (summary.dataset == "juliet") & ~summary.config.str.contains("CAP")].copy()
    d["gb"] = d.weights_mb / 1024
    d["name"] = d.config.str.replace("qwen7b_", "", regex=False)
    uniform = {"fp16": "FP16", "int8": "INT8", "nf4": "NF4", "hqq_4bit": "HQQ-4", "hqq_3bit": "HQQ-3", "hqq_2bit": "HQQ-2"}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    mixed48 = d[~d.name.isin(uniform) & ~d.name.str.startswith("hqq")]
    mixed2 = d[d.name.str.startswith("hqq_2bit_")]
    ax.scatter(mixed48.gb, mixed48.auroc, s=14, color=MUTED, label="INT8/NF4 mixes", zorder=2)
    ax.scatter(mixed2.gb, mixed2.auroc, s=14, color=ORANGE, label="2/4-bit mixes", zorder=2)
    u = d[d.name.isin(uniform)]
    ax.errorbar(u.gb, u.auroc, yerr=[u.auroc - u.auroc_lo, u.auroc_hi - u.auroc], fmt="o", ms=4,
                color=BLUE, ecolor=BLUE, elinewidth=0.8, capsize=0, label="Uniform", zorder=3)
    for _, r in u.iterrows():
        off = {"hqq_3bit": (-8, -14), "nf4": (2, -14), "hqq_4bit": (5, 6), "hqq_2bit": (5, -3)}.get(r["name"], (5, 3))
        ax.annotate(uniform[r["name"]], (r.gb, r.auroc), xytext=off, textcoords="offset points",
                    fontsize=7, color=INK, ha="right" if r["name"] == "hqq_3bit" else "left")
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Weight memory (GB)")
    ax.set_ylabel("Test AUROC")
    ax.legend(loc="lower right", fontsize=6.5)
    fig.savefig(f"{OUT}/pareto.pdf")
    plt.close(fig)


def lowbit_selection(summary):
    d = summary[(summary.model == "qwen7b") & (summary.dataset == "juliet")].set_index("config")
    named = [("qwen7b_hqq_2bit_last7", "Last 7"), ("qwen7b_hqq_2bit_top_gen", "Top by $S_{gen}$"),
             ("qwen7b_hqq_2bit_first7", "First 7"), ("qwen7b_hqq_2bit_top_task", "Top by $S_{task}$"),
             ("qwen7b_hqq_2bit_top_up", "Top by $S_{up}$")]
    named = [(k, v) for k, v in named if k in d.index]
    rand = d[d.index.str.startswith("qwen7b_hqq_2bit_rand_")].auroc.to_numpy()
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ys = np.arange(len(named) + 1)
    jitter = np.linspace(-0.12, 0.12, len(rand)) if len(rand) > 1 else [0]
    ax.scatter(rand, ys[0] + np.array(jitter), s=14, color=MUTED, zorder=3)
    ax.plot([rand.mean()] * 2, [ys[0] - 0.3, ys[0] + 0.3], color=INK2, lw=1.2)
    for y, (k, _) in zip(ys[1:], named):
        ax.scatter(d.loc[k, "auroc"], y, s=22, color=ORANGE if "top_up" in k else BLUE, zorder=3)
    ax.axvline(d.loc["qwen7b_fp16", "auroc"], color=AQUA, lw=1, ls="--")
    ax.axvline(d.loc["qwen7b_hqq_2bit", "auroc"], color=INK2, lw=1, ls=":")
    ax.text(d.loc["qwen7b_fp16", "auroc"], ys[-1] + 0.6, "FP16", color=INK, fontsize=7, ha="center")
    ax.text(d.loc["qwen7b_hqq_2bit", "auroc"], ys[-1] + 0.6, "uniform 2-bit", color=INK, fontsize=7, ha="center")
    ax.set_yticks(ys, [f"Random (n={len(rand)})"] + [v for _, v in named])
    ax.set_ylim(-0.6, ys[-1] + 1.0)
    ax.set_xlabel("Test AUROC (7 of 28 layers at 4-bit, rest 2-bit)")
    ax.grid(axis="y", visible=False)
    fig.savefig(f"{OUT}/lowbit_selection.pdf")
    plt.close(fig)


def sensitivity_depth():
    s4 = pd.read_csv("results/sensitivity_qwen7b_juliet.csv")
    s2 = pd.read_csv("results/sensitivity_qwen7b_hqq2_juliet.csv")
    panels = [(s2, "S_gen", "$S_{gen}$ (2-bit)"), (s2, "S_task", "$S_{task}$ (2-bit)"), (s4, "S_l2", "$S_{L2}$")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.7), sharex=True)
    for ax, (df, col, title) in zip(axes, panels):
        ax.bar(df.layer, df[col], width=0.8, color=BLUE)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("Layer")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    fig.tight_layout()
    fig.savefig(f"{OUT}/sensitivity_depth.pdf")
    plt.close(fig)


def score_shift():
    runs = [("qwen7b_juliet_qwen7b_fp16_test_cap_off", "FP16", AQUA),
            ("qwen7b_juliet_qwen7b_nf4_test_cap_off", "NF4", BLUE),
            ("qwen7b_juliet_qwen7b_hqq_2bit_test_cap_off", "HQQ-2", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8), sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, target, title in zip(axes, [0, 1], ["Safe functions", "Vulnerable functions"]):
        for rid, lab, col in runs:
            s = load_scores(rid)
            ax.hist(s.loc[s.target == target, "p_vuln"], bins=bins, histtype="step", lw=1.2, color=col, label=lab)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("$P(\\mathrm{vuln})$")
    axes[0].set_ylabel("Functions")
    axes[0].legend(fontsize=6.5, loc="upper center")
    fig.tight_layout()
    fig.savefig(f"{OUT}/score_shift.pdf")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = pd.read_csv("paper/summary.csv")
    pareto(summary)
    lowbit_selection(summary)
    sensitivity_depth()
    score_shift()
    print(f"Figures written to {OUT}/")


if __name__ == "__main__":
    main()
