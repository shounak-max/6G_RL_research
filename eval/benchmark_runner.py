"""
eval/benchmark_runner.py
========================
Standardized evaluation benchmark runner for 6G RRM agents and environments.

Runs deterministic or stochastic evaluation rollouts across multiple random seeds,
aggregates network KPIs and RL metrics via MetricsTracker, and outputs
clean formatted summary tables + optional JSON / CSV exports.

Usage
-----
    # Random baseline evaluation
    python -m eval.benchmark_runner --env base_rrm --num-episodes 20

    # Evaluate trained SB3 model
    python -m eval.benchmark_runner --env base_rrm --model-path runs/baseline_ppo/model.zip --num-episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import gymnasium as gym

from eval.metrics import MetricsTracker, timer_benchmark


def make_env(env_name: str, seed: int = 42, **kwargs) -> gym.Env:
    """Instantiate the target environment by short name."""
    if env_name in ["base_rrm", "rrm", "BaseRRMEnv"]:
        from envs.base_rrm_env import BaseRRMEnv, RRMEnvConfig
        cfg = RRMEnvConfig(**kwargs) if kwargs else RRMEnvConfig()
        env = BaseRRMEnv(config=cfg)
    elif env_name in ["uav_routing", "uav", "UAVRoutingEnv"]:
        from envs.uav_routing_env import UAVRoutingEnv, UAVEnvConfig
        cfg = UAVEnvConfig(**kwargs) if kwargs else UAVEnvConfig()
        env = UAVRoutingEnv(config=cfg)
    elif env_name in ["ris_phase", "ris", "RISPhaseEnv"]:
        from envs.ris_phase_env import RISPhaseEnv, RISConfig
        cfg = RISConfig(**kwargs) if kwargs else RISConfig()
        env = RISPhaseEnv(config=cfg)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
    return env


def evaluate_policy(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray], Any],
    num_episodes: int = 20,
    seed: int = 42,
    deterministic: bool = True,
) -> MetricsTracker:
    """
    Run evaluation rollouts using policy_fn on env.
    policy_fn takes obs and returns action.
    """
    tracker = MetricsTracker(name="eval_benchmark")

    for ep in range(num_episodes):
        ep_seed = seed + ep * 1000
        obs, info = env.reset(seed=ep_seed)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, lat_ms = timer_benchmark(policy_fn, obs)
            obs, reward, terminated, truncated, step_info = env.step(action)

            # Extract telecommunication metrics from info dict
            # 1) Throughput
            if "throughput_bits" in step_info:
                # 1 ms step -> Mbps
                tp = (step_info["throughput_bits"] / 1e6) / 1e-3
            else:
                tp = step_info.get("mean_throughput_mbps", step_info.get("throughput_mbps", 0.0))

            # 2) Delay
            if "mean_delay_ms" in step_info:
                delay = float(step_info["mean_delay_ms"])
            elif "mean_queue" in step_info:
                # Queue backlog in ms assuming ~5 packets/step service
                delay = float(step_info["mean_queue"]) * 0.2
            else:
                delay = step_info.get("delay_ms", 0.0)

            # 3) PDR
            pdr = step_info.get("pdr", step_info.get("packet_delivery_ratio", 1.0))

            # 4) Energy
            if "total_power_w" in step_info:
                # 1 ms step -> Joules
                energy = float(step_info["total_power_w"]) * 1e-3
            else:
                energy = step_info.get("energy_joules", step_info.get("power_consumed", 0.0))

            # 5) SLA violation
            sla_viols = step_info.get("sla_violations", step_info.get("sla_violation", False))
            sla_violated = (sla_viols > 0) if isinstance(sla_viols, (int, float)) else bool(sla_viols)

            comm_bytes = step_info.get("comm_bytes", 0)

            tracker.record_step(
                reward=reward,
                throughput_mbps=tp,
                delay_ms=delay,
                pdr=pdr,
                energy_joules=energy,
                sla_violated=sla_violated,
                comm_bytes=comm_bytes,
                inference_latency_ms=lat_ms,
            )

        tracker.end_episode()

    return tracker


def print_summary_table(stats: Dict[str, Any], title: str = "Benchmark Evaluation Results") -> None:
    """Print clean ASCII summary table of benchmark results."""
    print("\n" + "=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70)
    print(f"Total Episodes Evaluated : {stats.get('num_episodes', 0)}")
    print(f"Total Simulation Steps   : {stats.get('total_steps', 0)}")
    print("-" * 70)
    print(f"{'Metric':<36} | {'Mean ± 95% CI':<20} | {'Std':<8}")
    print("-" * 70)

    rows = [
        ("Total Reward", "total_reward"),
        ("Mean Step Reward", "mean_reward"),
        ("Mean Throughput (Mbps)", "mean_throughput_mbps"),
        ("Mean Delay (ms)", "mean_delay_ms"),
        ("Packet Delivery Ratio (PDR)", "mean_pdr"),
        ("Total Energy (Joules)", "total_energy_joules"),
        ("Energy Efficiency (Mbps/J)", "energy_efficiency_mbps_per_joule"),
        ("SLA Violation Rate", "sla_violation_rate"),
        ("Comm Overhead (Bytes)", "total_comm_bytes"),
        ("Inference Latency (ms)", "mean_inference_latency_ms"),
    ]

    for label, key in rows:
        mean_val = stats.get(f"{key}_mean", 0.0)
        ci_val = stats.get(f"{key}_ci95", 0.0)
        std_val = stats.get(f"{key}_std", 0.0)
        if "pdr" in key or "rate" in key:
            val_str = f"{mean_val*100:6.2f}% ± {ci_val*100:5.2f}%"
        else:
            val_str = f"{mean_val:10.4f} ± {ci_val:8.4f}"
        print(f"{label:<36} | {val_str:<20} | {std_val:8.4f}")

    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="6G RRM Evaluation Benchmark Runner")
    parser.add_argument("--env", type=str, default="base_rrm", choices=["base_rrm", "uav_routing", "ris_phase"], help="Environment name")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model .zip (optional)")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save output JSON metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[*] Initializing benchmark on '{args.env}' ({args.num_episodes} episodes, seed={args.seed})...")
    env = make_env(args.env, seed=args.seed)

    if args.model_path and os.path.exists(args.model_path):
        print(f"[*] Loading model from {args.model_path}...")
        try:
            from stable_baselines3 import PPO, DQN
            try:
                model = PPO.load(args.model_path)
            except Exception:
                model = DQN.load(args.model_path)

            def policy_fn(obs):
                action, _ = model.predict(obs, deterministic=True)
                return action
        except Exception as e:
            print(f"[!] Failed to load model via SB3: {e}. Falling back to random action policy.")
            def policy_fn(obs):
                return env.action_space.sample()
    else:
        print("[*] No model path specified. Evaluating Random Action Policy.")
        def policy_fn(obs):
            return env.action_space.sample()

    tracker = evaluate_policy(env, policy_fn, num_episodes=args.num_episodes, seed=args.seed)
    stats = tracker.aggregate_statistics()
    print_summary_table(stats, title=f"Benchmark: {args.env.upper()} ({'Random' if not args.model_path else 'Trained Agent'})")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[+] Saved evaluation metrics to {args.output_json}")

    env.close()


if __name__ == "__main__":
    main()
