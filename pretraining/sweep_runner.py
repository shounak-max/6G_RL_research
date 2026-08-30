"""
pretraining/sweep_runner.py
===========================
Parallel SLA Weight Vector Sweep Runner (Phase 1).

Generates diverse SLA priority weight vectors (throughput, delay, energy, PDR),
trains expert policies on the target environment, evaluates convergence,
and registers all models in the ExpertPolicyDB.

Reference: Roadmap §3 Phase 1 (Sweep runner)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml
import gymnasium as gym

from envs.base_rrm_env import BaseRRMEnv, RRMEnvConfig, SLAWeights
from envs.uav_routing_env import UAVRoutingEnv, UAVEnvConfig, UAVSLAWeights
from pretraining.expert_policy_db import ExpertPolicyDB


def generate_sla_weight_vectors(num_vectors: int = 16, seed: int = 42) -> List[List[float]]:
    """
    Generate diverse normalized SLA weight vectors w = [w_tp, w_delay, w_energy, w_pdr]
    using Dirichlet distribution sampling + boundary extreme points.
    """
    rng = np.random.default_rng(seed)
    # Include extreme corner profiles (Throughput-first, Delay-first, Energy-first, Reliability-first)
    corners = [
        [0.85, 0.05, 0.05, 0.05],
        [0.05, 0.85, 0.05, 0.05],
        [0.05, 0.05, 0.85, 0.05],
        [0.05, 0.05, 0.05, 0.85],
        [0.25, 0.25, 0.25, 0.25],
    ]

    vectors = list(corners)
    remaining = num_vectors - len(vectors)
    if remaining > 0:
        # Sample from symmetric Dirichlet
        dirichlet_samples = rng.dirichlet(alpha=[1.0, 1.0, 1.0, 1.0], size=remaining)
        vectors.extend(dirichlet_samples.tolist())

    return vectors[:num_vectors]


class SLASweepRunner:
    """Orchestrates SLA weight sweep and expert policy database generation."""

    def __init__(
        self,
        env_name: str = "base_rrm",
        db_dir: str = "data/expert_policies",
        output_dir: str = "runs/sweep",
    ) -> None:
        self.env_name = env_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = ExpertPolicyDB(db_dir=db_dir)

    def run_sweep(
        self,
        weight_vectors: List[List[float]],
        timesteps_per_vector: int = 10000,
        base_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Execute training sweep across weight vectors.
        """
        print("=" * 75)
        print(f"Starting SLA Pretraining Sweep on '{self.env_name}'")
        print(f"Total Weight Vectors : {len(weight_vectors)}")
        print(f"Timesteps / Vector   : {timesteps_per_vector:,}")
        print(f"Database Directory   : {self.db.db_dir}")
        print("=" * 75)

        results = []

        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise RuntimeError("Stable-Baselines3 is required to execute the SLA sweep.")

        for idx, w in enumerate(weight_vectors):
            t0 = time.perf_counter()
            policy_id = f"{self.env_name}_expert_w{idx:03d}"
            print(f"\n[{idx+1}/{len(weight_vectors)}] Training Expert '{policy_id}' | Weights: {[round(x, 3) for x in w]}...")

            # Instantiate env with target SLA
            if self.env_name in ["base_rrm", "rrm"]:
                sla = SLAWeights(throughput=w[0], delay=w[1], energy=w[2], pdr=w[3])
                env = BaseRRMEnv(RRMEnvConfig(sla=sla))
            elif self.env_name in ["uav_routing", "uav"]:
                sla = UAVSLAWeights(delay=w[0], energy=w[1], link=w[2], progress=w[3])
                env = UAVRoutingEnv(UAVEnvConfig(sla=sla))
            else:
                raise ValueError(f"Unsupported environment: {self.env_name}")

            # Train PPO agent
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=5e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=5,
                gamma=0.99,
                seed=base_seed + idx,
                verbose=0,
            )

            model.learn(total_timesteps=timesteps_per_vector)
            train_duration = time.perf_counter() - t0

            # Checkpoint
            chk_path = self.output_dir / f"{policy_id}.zip"
            model.save(str(chk_path))

            # Evaluation & Trajectory Feature Extraction
            eval_rewards = []
            eval_pdr = []
            eval_tp = []
            eval_delay = []
            state_samples = []

            obs, _ = env.reset(seed=base_seed + 1000 + idx)
            for _ in range(200):
                act, _ = model.predict(obs, deterministic=True)
                state_samples.append(obs)
                obs, r, term, trunc, info = env.step(act)
                eval_rewards.append(r)
                eval_pdr.append(info.get("pdr", 1.0))
                eval_tp.append(info.get("mean_throughput_mbps", info.get("throughput_bits", 0.0) / 1000.0))
                eval_delay.append(info.get("mean_delay_ms", 5.0))
                if term or trunc:
                    obs, _ = env.reset()

            state_mat = np.array(state_samples)
            feat_mean = state_mat.mean(axis=0).tolist() if len(state_mat) > 0 else []
            feat_std = state_mat.std(axis=0).tolist() if len(state_mat) > 0 else []

            # Register in Database
            entry = self.db.add_expert(
                policy_id=policy_id,
                env_name=self.env_name,
                weight_vector=w,
                model_file=str(chk_path),
                total_timesteps=timesteps_per_vector,
                final_reward=float(np.mean(eval_rewards)),
                convergence_steps=timesteps_per_vector,
                mean_throughput_mbps=float(np.mean(eval_tp)),
                mean_delay_ms=float(np.mean(eval_delay)),
                pdr=float(np.mean(eval_pdr)),
                sla_violation_rate=float(1.0 - np.mean(eval_pdr)),
                state_feature_mean=feat_mean,
                state_feature_std=feat_std,
                metadata={"train_duration_sec": train_duration},
            )

            print(f"    -> Done in {train_duration:.1f}s | Mean Eval Reward: {entry.final_reward:.4f} | PDR: {entry.pdr*100:.1f}%")
            results.append(entry.to_dict())
            env.close()

        print(f"\n[+] Pretraining Sweep Completed! Total Experts in DB: {len(self.db)}")
        return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run SLA Weight Pretraining Sweep")
    parser.add_argument("--env", type=str, default="base_rrm", choices=["base_rrm", "uav_routing"])
    parser.add_argument("--num-vectors", type=int, default=8, help="Number of SLA weight vectors to sweep")
    parser.add_argument("--timesteps-per-vector", type=int, default=5000, help="Timesteps per expert policy")
    parser.add_argument("--db-dir", type=str, default="data/expert_policies", help="Expert Policy DB path")
    parser.add_argument("--output-dir", type=str, default="runs/sweep", help="Sweep output directory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    vectors = generate_sla_weight_vectors(num_vectors=args.num_vectors, seed=args.seed)
    runner = SLASweepRunner(env_name=args.env, db_dir=args.db_dir, output_dir=args.output_dir)
    runner.run_sweep(weight_vectors=vectors, timesteps_per_vector=args.timesteps_per_vector, base_seed=args.seed)


if __name__ == "__main__":
    main()
