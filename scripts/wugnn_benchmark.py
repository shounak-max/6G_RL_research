"""
scripts/wugnn_benchmark.py
==========================
WUGNN vs. Classical WMMSE Execution-Time Benchmark (Phase 4 / KDDL).

Generates:
  figures/fig5_wugnn_speedup.png -- Latency & speedup ratio across N transceivers.

Usage
-----
python scripts/wugnn_benchmark.py
python scripts/wugnn_benchmark.py --n-list 10 20 50 100 --n-repeats 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.kddl.knowledge_embedded.wugnn import WUGNNBenchmark


def generate_speedup_figure(results: dict, output_path: str = "figures/fig5_wugnn_speedup.png") -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    Ns = sorted(results["wmmse_mean_s"].keys())
    wmmse_ms = [results["wmmse_mean_s"][N] * 1000 for N in Ns]
    wugnn_ms = [results["wugnn_mean_s"][N] * 1000 for N in Ns]
    speedups = [results["speedup"][N] for N in Ns]

    fig = plt.figure(figsize=(14, 5), facecolor="#0d1117")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    GOLD = "#f5c518"
    TEAL = "#00e5ff"
    PURPLE = "#ce93d8"
    GRID = "#1e2a3a"

    for ax in [ax1, ax2]:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white", labelsize=9)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["left"].set_color(GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.6, linestyle="--")

    x = np.arange(len(Ns))
    w = 0.35
    bars1 = ax1.bar(x - w/2, wmmse_ms, w, color=GOLD, alpha=0.85, label="Classical WMMSE")
    bars2 = ax1.bar(x + w/2, wugnn_ms, w, color=TEAL, alpha=0.85, label="WUGNN (GNN)")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N={n}" for n in Ns])
    ax1.set_xlabel("Number of Transceivers N", color="white", fontsize=10)
    ax1.set_ylabel("Mean Inference Time (ms, log scale)", color="white", fontsize=10)
    ax1.set_title("Inference Latency Comparison", color="white", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, facecolor="#1a2233", edgecolor=GRID, labelcolor="white")
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h * 1.12, f"{h:.2f}", ha="center", va="bottom", color=GOLD, fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h * 1.12, f"{h:.3f}", ha="center", va="bottom", color=TEAL, fontsize=7)

    ax2.plot(Ns, speedups, color=PURPLE, linewidth=2.5, marker="D", markersize=8, label="Speedup ratio")
    ax2.fill_between(Ns, speedups, alpha=0.15, color=PURPLE)
    ax2.axhline(100.0, color=GOLD, linewidth=1.2, linestyle=":", alpha=0.7, label="100× target")
    for ni, (N, s) in enumerate(zip(Ns, speedups)):
        ax2.annotate(f"{s:.0f}×", xy=(N, s), xytext=(0, 10), textcoords="offset points",
                     ha="center", color="white", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Number of Transceivers N", color="white", fontsize=10)
    ax2.set_ylabel("Speedup (WMMSE / WUGNN)", color="white", fontsize=10)
    ax2.set_title("WUGNN Execution Speedup vs. WMMSE", color="white", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, facecolor="#1a2233", edgecolor=GRID, labelcolor="white")
    ax2.set_yscale("log")

    fig.suptitle(
        "KDDL Phase 4: WMMSE-Unrolled GNN — Execution Time Benchmark",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] WUGNN speedup figure saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="WUGNN vs WMMSE Execution-Time Benchmark")
    parser.add_argument("--n-list", type=int, nargs="+", default=[10, 20, 50, 100], help="Transceiver counts to benchmark")
    parser.add_argument("--n-repeats", type=int, default=20, help="Timing repetitions per N")
    parser.add_argument("--wmmse-iters", type=int, default=100, help="WMMSE iterations (classical solver)")
    parser.add_argument("--wugnn-layers", type=int, default=8, help="WUGNN unrolled GNN layers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=str, default="data/wugnn_benchmark_results.json")
    parser.add_argument("--figure-path", type=str, default="figures/fig5_wugnn_speedup.png")
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark = WUGNNBenchmark(
        transceiver_counts=args.n_list,
        n_repeats=args.n_repeats,
        wmmse_iters=args.wmmse_iters,
        wugnn_layers=args.wugnn_layers,
        seed=args.seed,
    )
    results = benchmark.run()

    Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "wmmse_mean_s": {str(k): v for k, v in results["wmmse_mean_s"].items()},
        "wugnn_mean_s": {str(k): v for k, v in results["wugnn_mean_s"].items()},
        "speedup": {str(k): v for k, v in results["speedup"].items()},
    }
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"[+] Benchmark results saved to: {args.save_json}")

    if not args.no_figure:
        try:
            generate_speedup_figure(results, output_path=args.figure_path)
        except Exception as e:
            print(f"[!] Figure generation failed: {e}")


if __name__ == "__main__":
    main()
