"""
agents/marl/marl_ablation_runner.py
===================================
MARL Ablation & Robustness Benchmarking Harness (Phase 2).

Directly compares 3 communication paradigms across clean and blockage-affected channels:
1. Independent Learners (No communication — 0 Bytes)
2. Full Broadcast Sharing (All-to-all communication — O(N^2) Bytes)
3. GAT-CRL Selective Sharing (Top-k bounded communication — O(kN) Bytes)

Evaluates:
- Convergence reward & variance (non-stationarity mitigation)
- Signaling overhead scaling (sub-linear verification)
- Fault-tolerance under sudden mmWave/THz blockage dropouts (Reviewer Note)

Usage:
    python agents/marl/marl_ablation_runner.py --steps 1000 --num-ues 8
    python agents/marl/marl_ablation_runner.py --include-blockage --steps 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from agents.marl.independent_learners import run_independent_baseline
from agents.marl.gat_crl.gat_crl_trainer import train_gat_crl


def _run_scheme_multi_seed(
    scheme_fn,
    scheme_kwargs: dict,
    num_episodes: int,
    base_seed: int,
) -> Dict[str, Any]:
    """
    Run scheme_fn across num_episodes independent random seeds.
    Returns aggregated statistics: mean, std, 95% CI half-width for each metric.
    """
    rewards, variances, comm_bytes_list = [], [], []
    pdrs, sla_viols = [], []

    for ep in range(num_episodes):
        seed = base_seed + ep * 97  # Different seed per episode
        kwargs = {**scheme_kwargs, "seed": seed}
        r = scheme_fn(**kwargs)
        rewards.append(r.get("late_reward", 0.0))
        variances.append(r.get("late_variance", 0.0))
        comm_bytes_list.append(r.get("mean_comm_bytes", 0.0))
        pdrs.append(r.get("pdr", r.get("late_reward", 0.0)))  # fallback to reward
        sla_viols.append(r.get("sla_violation_rate", 0.0))

    def _agg(vals):
        arr = np.array(vals)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        # 95% CI half-width: z=1.96 for normal approximation
        ci95 = 1.96 * std / np.sqrt(max(len(arr), 1))
        return {"mean": mean, "std": std, "ci95": round(ci95, 5)}

    return {
        "late_reward": float(np.mean(rewards)),
        "late_variance": float(np.mean(variances)),
        "mean_comm_bytes": float(np.mean(comm_bytes_list)),
        "stats": {
            "reward": _agg(rewards),
            "variance": _agg(variances),
            "comm_bytes": _agg(comm_bytes_list),
        },
        "num_seeds": num_episodes,
    }


def run_full_ablation(
    num_ues: int = 8,
    num_rbs: int = 12,
    num_steps: int = 1500,
    include_blockage: bool = True,
    seed: int = 42,
    output_json: Optional[str] = None,
    num_episodes: int = 1,
) -> Dict[str, Any]:
    """
    Run complete ablation matrix across schemes and channel conditions.
    With num_episodes > 1, each scheme is run across multiple seeds for 95% CI.
    """
    print("\n" + "=" * 80)
    print(f"PHASE 2 COOPERATIVE MARL (GAT-CRL) ABLATION SUITE ({num_ues} UEs, {num_steps} Steps, {num_episodes} seeds)")
    print("=" * 80)

    results: Dict[str, Any] = {}

    # 1. Clean Channel: Independent Learners
    print("\n[1/6] Running Independent Learners (Clean Channel)...")
    res_ind_clean = _run_scheme_multi_seed(
        run_independent_baseline,
        {"num_ues": num_ues, "num_rbs": num_rbs, "num_steps": num_steps},
        num_episodes=num_episodes,
        base_seed=seed,
    )
    results["independent_clean"] = {
        "scheme": "Independent Learners",
        "channel": "Clean",
        "late_reward": res_ind_clean["late_reward"],
        "late_variance": res_ind_clean["late_variance"],
        "comm_bytes_per_step": 0.0,
        "stats": res_ind_clean.get("stats", {}),
    }

    # 2. Clean Channel: Full Broadcast (top_k = num_ues - 1)
    print("\n[2/6] Running Full Broadcast MARL (Clean Channel)...")
    res_full_clean = _run_scheme_multi_seed(
        train_gat_crl,
        {"num_ues": num_ues, "num_rbs": num_rbs, "num_steps": num_steps,
         "top_k": num_ues - 1, "realistic_channel": False},
        num_episodes=num_episodes,
        base_seed=seed,
    )
    results["full_broadcast_clean"] = {
        "scheme": "Full Broadcast",
        "channel": "Clean",
        "late_reward": res_full_clean["late_reward"],
        "late_variance": res_full_clean["late_variance"],
        "comm_bytes_per_step": res_full_clean["mean_comm_bytes"],
        "stats": res_full_clean.get("stats", {}),
    }

    # 3. Clean Channel: GAT-CRL Selective Sharing (top_k = 2)
    print("\n[3/6] Running GAT-CRL Selective Sharing (top-k=2, Clean Channel)...")
    res_gat_clean = _run_scheme_multi_seed(
        train_gat_crl,
        {"num_ues": num_ues, "num_rbs": num_rbs, "num_steps": num_steps,
         "top_k": 2, "realistic_channel": False},
        num_episodes=num_episodes,
        base_seed=seed,
    )
    results["gat_crl_clean"] = {
        "scheme": "GAT-CRL (Selective)",
        "channel": "Clean",
        "late_reward": res_gat_clean["late_reward"],
        "late_variance": res_gat_clean["late_variance"],
        "comm_bytes_per_step": res_gat_clean["mean_comm_bytes"],
        "stats": res_gat_clean.get("stats", {}),
    }

    if include_blockage:
        # 4. Blockage Channel: Independent Learners
        print("\n[4/6] Running Independent Learners (Blockage Channel)...")
        res_ind_block = _run_scheme_multi_seed(
            run_independent_baseline,
            {"num_ues": num_ues, "num_rbs": num_rbs, "num_steps": num_steps},
            num_episodes=num_episodes,
            base_seed=seed + 1000,
        )
        results["independent_blockage"] = {
            "scheme": "Independent Learners",
            "channel": "Realistic Blockage",
            "late_reward": res_ind_block["late_reward"],
            "late_variance": res_ind_block["late_variance"],
            "comm_bytes_per_step": 0.0,
            "stats": res_ind_block.get("stats", {}),
        }

        # 5. Blockage Channel: Full Broadcast
        print("\n[5/6] Running Full Broadcast MARL (Blockage Channel)...")
        res_full_block = _run_scheme_multi_seed(
            train_gat_crl,
            {"num_ues": num_ues, "num_rbs": num_rbs, "num_steps": num_steps,
             "top_k": num_ues - 1, "realistic_channel": True},
            num_episodes=num_episodes,
            base_seed=seed + 1000,
        )
        results["full_broadcast_blockage"] = {
            "scheme": "Full Broadcast",
            "channel": "Realistic Blockage",
            "late_reward": res_full_block["late_reward"],
            "late_variance": res_full_block["late_variance"],
            "comm_bytes_per_step": res_full_block["mean_comm_bytes"],
            "stats": res_full_block.get("stats", {}),
        }

        # 6. Blockage Channel: GAT-CRL Selective Sharing
        print("\n[6/6] Running GAT-CRL Selective Sharing (Blockage Channel)...")
        res_gat_block = _run_scheme_multi_seed(
            train_gat_crl,
            {"num_ues": num_ues, "num_rbs": num_rbs, "num_steps": num_steps,
             "top_k": 2, "realistic_channel": True},
            num_episodes=num_episodes,
            base_seed=seed + 1000,
        )
        results["gat_crl_blockage"] = {
            "scheme": "GAT-CRL (Selective)",
            "channel": "Realistic Blockage",
            "late_reward": res_gat_block["late_reward"],
            "late_variance": res_gat_block["late_variance"],
            "comm_bytes_per_step": res_gat_block["mean_comm_bytes"],
            "stats": res_gat_block.get("stats", {}),
        }

    # Print Ablation Summary Table
    print("\n" + "=" * 85)
    print(f"{'PHASE 2 MARL ABLATION SUMMARY MATRIX':^85}")
    print("=" * 85)
    print(f"{'Scheme':<26} | {'Channel Mode':<18} | {'Late Reward':<12} | {'Variance':<10} | {'Comm Overhead':<14}")
    print("-" * 85)

    for k, v in results.items():
        print(
            f"{v['scheme']:<26} | {v['channel']:<18} | "
            f"{v['late_reward']:<12.4f} | {v['late_variance']:<10.5f} | "
            f"{v['comm_bytes_per_step']:<8.0f} B/step"
        )
    print("=" * 85 + "\n")

    # Signaling overhead reduction comparison
    clean_full_bytes = results["full_broadcast_clean"]["comm_bytes_per_step"]
    clean_gat_bytes = results["gat_crl_clean"]["comm_bytes_per_step"]
    overhead_reduction = (1.0 - clean_gat_bytes / max(1.0, clean_full_bytes)) * 100.0
    print(f"[+] GAT-CRL Signaling Overhead Reduction vs. Full Broadcast: {overhead_reduction:.1f}% saved per step.")

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[+] Saved ablation results to {output_json}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run MARL Ablation Benchmark")
    parser.add_argument("--num-ues", type=int, default=8, help="Number of UEs")
    parser.add_argument("--num-rbs", type=int, default=12, help="Number of RBs")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per run")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of independent seeds per scheme (for 95%% CI error bars). Default 1.")
    parser.add_argument("--include-blockage", action="store_true", default=True, help="Include blockage channel ablation")
    parser.add_argument("--output-json", type=str, default="data/marl_ablation_results.json", help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    run_full_ablation(
        num_ues=args.num_ues,
        num_rbs=args.num_rbs,
        num_steps=args.steps,
        include_blockage=args.include_blockage,
        seed=args.seed,
        output_json=args.output_json,
        num_episodes=args.num_episodes,
    )


if __name__ == "__main__":
    main()
