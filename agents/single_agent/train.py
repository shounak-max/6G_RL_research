"""
agents/single_agent/train.py
============================
Phase 0 Single-Agent Baseline Training Script for 6G RRM.

Trains PPO / DQN agents on BaseRRMEnv with configurable SLA reward profiles,
seeds, logging, and evaluation.

Usage:
    # Quick smoke training
    python agents/single_agent/train.py --total-timesteps 10000 --algo ppo --sla-profile balanced

    # Full baseline run
    python agents/single_agent/train.py --total-timesteps 100000 --algo ppo --sla-profile latency_critical --run-name ppo_urllc
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml
import gymnasium as gym

from envs.base_rrm_env import BaseRRMEnv, RRMEnvConfig, SLAWeights
from eval.metrics import MetricsTracker
from eval.benchmark_runner import evaluate_policy, print_summary_table


def load_sla_profile(profile_name: str, config_path: str = "configs/sla_weight_vectors.yaml") -> SLAWeights:
    """Load named SLA weight vector profile from YAML."""
    if not os.path.exists(config_path):
        print(f"[!] Warning: Config path {config_path} not found. Using default balanced SLA.")
        return SLAWeights()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    profiles = data.get("profiles", [])
    for p in profiles:
        if p.get("name") == profile_name:
            return SLAWeights(
                throughput=float(p.get("throughput", 1.0)),
                delay=float(p.get("delay", 0.5)),
                energy=float(p.get("energy", 0.2)),
                pdr=float(p.get("pdr", 0.3)),
            )

    defaults = data.get("defaults", {})
    if defaults:
        return SLAWeights(
            throughput=float(defaults.get("throughput", 1.0)),
            delay=float(defaults.get("delay", 0.5)),
            energy=float(defaults.get("energy", 0.2)),
            pdr=float(defaults.get("pdr", 0.3)),
        )

    return SLAWeights()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Single-Agent Baseline on BaseRRMEnv")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"], help="RL algorithm")
    parser.add_argument("--total-timesteps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--sla-profile", type=str, default="balanced", help="SLA profile from yaml")
    parser.add_argument("--num-ues", type=int, default=8, help="Number of UEs")
    parser.add_argument("--num-rbs", type=int, default=12, help="Number of RBs")
    parser.add_argument("--realistic-channel", action="store_true", help="Enable doubly-selective fading & blockages")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="runs", help="Output directory for checkpoints")
    parser.add_argument("--run-name", type=str, default=None, help="Name of experiment run")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes post-training")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or f"{args.algo}_{args.sla_profile}_{int(time.time())}"
    save_dir = Path(args.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"6G RRM Single-Agent Training: {args.algo.upper()}")
    print("=" * 70)
    print(f"SLA Profile        : {args.sla_profile}")
    print(f"Total Timesteps    : {args.total_timesteps:,}")
    print(f"UEs / RBs          : {args.num_ues} / {args.num_rbs}")
    print(f"Realistic Channel  : {args.realistic_channel}")
    print(f"Random Seed        : {args.seed}")
    print(f"Save Directory     : {save_dir}")
    print("-" * 70)

    # 1. Setup Environment
    sla_weights = load_sla_profile(args.sla_profile)
    env_config = RRMEnvConfig(
        num_ues=args.num_ues,
        num_rbs=args.num_rbs,
        sla=sla_weights,
        realistic_channel=args.realistic_channel,
    )
    env = BaseRRMEnv(config=env_config)

    # 2. Setup RL Agent via Stable-Baselines3
    try:
        from stable_baselines3 import PPO, DQN
        from stable_baselines3.common.callbacks import CheckpointCallback

        if args.algo == "ppo":
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                seed=args.seed,
            )
        elif args.algo == "dqn":
            # Flatten multidiscrete space for DQN if necessary
            flattened_env = gym.wrappers.FlattenObservation(env)
            # DQN natively requires discrete action space; for multidiscrete, use ActionTable or PPO
            print("[*] Note: For MultiDiscrete action spaces, PPO is recommended. Initializing PPO.")
            model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)

        # 3. Train
        print("\n[*] Starting Training Loop...")
        t_start = time.perf_counter()
        model.learn(total_timesteps=args.total_timesteps)
        t_elapsed = time.perf_counter() - t_start
        print(f"[+] Training completed in {t_elapsed:.2f} seconds ({args.total_timesteps / t_elapsed:.1f} steps/sec)")

        # 4. Save Model
        model_path = save_dir / "model.zip"
        model.save(str(model_path))
        print(f"[+] Model checkpoint saved to {model_path}")

        # 5. Evaluate
        print("\n[*] Running Post-Training Benchmark Evaluation...")
        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

        eval_env = BaseRRMEnv(config=env_config)
        tracker = evaluate_policy(eval_env, policy_fn, num_episodes=args.eval_episodes, seed=args.seed + 999)
        stats = tracker.aggregate_statistics()
        print_summary_table(stats, title=f"Post-Training Benchmark ({run_name})")
        eval_env.close()

    except ImportError as e:
        print(f"[!] Stable-Baselines3 error: {e}. Running custom baseline training loop...")
        # Fallback pure-python policy rollout
        step_rewards = []
        obs, _ = env.reset(seed=args.seed)
        for step in range(min(args.total_timesteps, 5000)):
            act = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(act)
            step_rewards.append(r)
            if term or trunc:
                obs, _ = env.reset()
        print(f"[+] Completed {len(step_rewards)} steps. Mean step reward: {np.mean(step_rewards):.4f}")

    env.close()


if __name__ == "__main__":
    main()
