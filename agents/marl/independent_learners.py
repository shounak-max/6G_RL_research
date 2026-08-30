"""
agents/marl/independent_learners.py
===================================
Independent Learners MARL Baseline (Phase 2).

Implements decentralized Independent Actor-Critic / Independent Q-Learning agents.
Each agent perceives only its local state and takes actions without communicating
with neighboring nodes.

Purpose:
- Acts as the primary non-stationarity baseline in Phase 2.
- Demonstrates performance degradation and variance inflation as agent count scales,
  directly justifying the need for GAT-CRL cooperative policy sharing.

Reference: Roadmap §3 Phase 2 (Independent learners baseline)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.base_rrm_env import BaseRRMEnv, RRMEnvConfig
from eval.metrics import MetricsTracker


class LocalActorCritic(nn.Module):
    """Local Actor-Critic network for an individual agent."""

    def __init__(self, local_obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(local_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.feature_net(x)
        logits = self.actor_head(feat)
        value = self.critic_head(feat)
        return logits, value


class IndependentLearnersBaseline:
    """
    Independent Decentralized MARL Baseline.
    Manages N decoupled Actor-Critic learners without coordination.
    """

    def __init__(
        self,
        num_agents: int = 8,
        local_obs_dim: int = 16,
        action_dim: int = 48,  # e.g., 12 RBs * 4 Power levels
        lr: float = 1e-3,
        gamma: float = 0.99,
        seed: int = 42,
    ) -> None:
        self.num_agents = num_agents
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.agents = [
            LocalActorCritic(local_obs_dim, action_dim)
            for _ in range(num_agents)
        ]
        self.optimizers = [
            optim.Adam(ag.parameters(), lr=lr)
            for ag in self.agents
        ]

    def select_actions(self, local_observations: List[np.ndarray]) -> Tuple[List[int], List[torch.Tensor], List[torch.Tensor]]:
        """Sample actions independently for each agent."""
        actions = []
        log_probs = []
        values = []

        for i, ag in enumerate(self.agents):
            obs_t = torch.tensor(local_observations[i], dtype=torch.float32).unsqueeze(0)
            logits, val = ag(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            log_prob = dist.log_prob(act)

            actions.append(int(act.item()))
            log_probs.append(log_prob)
            values.append(val.squeeze(0))

        return actions, log_probs, values

    def train_step(
        self,
        local_obs_list: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_obs_list: List[np.ndarray],
        dones: List[bool],
        log_probs: List[torch.Tensor],
        values: List[torch.Tensor],
    ) -> float:
        """Update each agent independently via Actor-Critic loss."""
        total_loss = 0.0

        for i, ag in enumerate(self.agents):
            next_obs_t = torch.tensor(next_obs_list[i], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, next_val = ag(next_obs_t)
                target_val = rewards[i] + (0.0 if dones[i] else self.gamma * next_val.item())

            advantage = target_val - values[i].item()
            actor_loss = -log_probs[i] * advantage
            critic_loss = nn.functional.mse_loss(values[i], torch.tensor([target_val], dtype=torch.float32))
            loss = actor_loss + 0.5 * critic_loss

            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

            total_loss += float(loss.item())

        return total_loss / self.num_agents

    def extract_local_obs(self, global_obs: np.ndarray, num_ues: int, num_rbs: int) -> List[np.ndarray]:
        """Split global RRM observation into per-UE local slices."""
        # channel_gains (num_ues * num_rbs) + sinrs + queue + pos
        gains = global_obs[:num_ues * num_rbs].reshape((num_ues, num_rbs))
        sinrs = global_obs[num_ues * num_rbs : 2 * num_ues * num_rbs].reshape((num_ues, num_rbs))
        queues = global_obs[2 * num_ues * num_rbs : 2 * num_ues * num_rbs + num_ues]
        pos = global_obs[2 * num_ues * num_rbs + num_ues :].reshape((num_ues, 2))

        local_obs = []
        for i in range(num_ues):
            local_vec = np.concatenate([
                gains[i],
                sinrs[i],
                [queues[i]],
                pos[i],
            ]).astype(np.float32)
            local_obs.append(local_vec)
        return local_obs


def run_independent_baseline(
    num_ues: int = 8,
    num_rbs: int = 12,
    num_steps: int = 3000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train independent learners on BaseRRMEnv."""
    print("=" * 70)
    print(f"Independent Learners Baseline (MARL - {num_ues} Agents)")
    print(f"Total Steps        : {num_steps:,}")
    print(f"Communication Cost : 0 Bytes (Independent / Decoupled)")
    print("=" * 70)

    env = BaseRRMEnv(RRMEnvConfig(num_ues=num_ues, num_rbs=num_rbs))
    local_dim = num_rbs * 2 + 1 + 2  # gains + sinrs + queue + (x,y)
    num_actions = num_rbs * 4        # 12 RBs * 4 power levels

    marl = IndependentLearnersBaseline(
        num_agents=num_ues,
        local_obs_dim=local_dim,
        action_dim=num_actions,
        seed=seed,
    )

    tracker = MetricsTracker(name="independent_marl")
    global_obs, _ = env.reset(seed=seed)
    ep_rewards = []
    step_rewards = []

    for step in range(num_steps):
        local_obs = marl.extract_local_obs(global_obs, num_ues, num_rbs)
        actions, log_probs, values = marl.select_actions(local_obs)

        # Convert discrete action indices to MultiDiscrete pairs
        multi_action = []
        for a in actions:
            rb_idx = a // 4
            pw_idx = a % 4
            multi_action.extend([rb_idx, pw_idx])

        next_global_obs, r, term, trunc, step_info = env.step(np.array(multi_action, dtype=np.int64))
        next_local_obs = marl.extract_local_obs(next_global_obs, num_ues, num_rbs)

        per_agent_reward = [r / num_ues] * num_ues
        dones = [term or trunc] * num_ues

        marl.train_step(local_obs, actions, per_agent_reward, next_local_obs, dones, log_probs, values)

        step_rewards.append(r)
        tracker.record_step(
            reward=r,
            throughput_mbps=step_info.get("mean_throughput_mbps", 0.0),
            delay_ms=step_info.get("mean_delay_ms", 0.0),
            pdr=step_info.get("pdr", 1.0),
            energy_joules=step_info.get("energy_joules", 0.0),
            sla_violated=step_info.get("sla_violations", 0) > 0,
            comm_bytes=0,
        )

        global_obs = next_global_obs
        if term or trunc:
            ep_summary = tracker.end_episode()
            ep_rewards.append(ep_summary.total_reward)
            global_obs, _ = env.reset()

    if tracker._current_steps:
        tracker.end_episode()

    stats = tracker.aggregate_statistics()
    # Compute reward stability / variance across windows
    early_rew = np.mean(step_rewards[:500]) if len(step_rewards) >= 500 else np.mean(step_rewards)
    late_rew = np.mean(step_rewards[-500:]) if len(step_rewards) >= 500 else np.mean(step_rewards)
    variance = float(np.var(step_rewards[-500:])) if len(step_rewards) >= 500 else float(np.var(step_rewards))

    print(f"\n[+] Completed {num_steps} steps.")
    print(f"    Early Reward (first 500 steps) : {early_rew:.4f}")
    print(f"    Late Reward  (last 500 steps)  : {late_rew:.4f}")
    print(f"    Late Convergence Variance      : {variance:.5f} (Non-stationarity noise)")

    env.close()
    return {
        "stats": stats,
        "early_reward": early_rew,
        "late_reward": late_rew,
        "late_variance": variance,
        "step_rewards": step_rewards,
    }


if __name__ == "__main__":
    run_independent_baseline(num_steps=1000)
