"""
agents/marl/gat_crl/gat_crl_trainer.py
======================================
End-to-End GAT-CRL Multi-Agent Training and Coordination Loop (Phase 2).

Integrates:
- GraphTopology: exports spatial-temporal transceiver interference graphs
- GATEncoder: extracts multi-head attention node embeddings
- NeuralTensorNetworkSimilarity: scores bilateral agent interactions
- SelectivePolicySharing: exchanges and fuses policy distributions under top-k/bandwidth constraints
- Decentralized Actor-Critic optimization

Reference: Roadmap §3 Phase 2 (GAT-CRL)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from envs.base_rrm_env import BaseRRMEnv, RRMEnvConfig, SLAWeights
from envs.graph_topology import GraphTopology
from agents.marl.gat_crl.gat_encoder import GATEncoder
from agents.marl.gat_crl.ntn_similarity import NeuralTensorNetworkSimilarity
from agents.marl.gat_crl.selective_sharing import CommunicationMetrics, SelectivePolicySharing
from eval.metrics import MetricsTracker


class GATCRLAgent(nn.Module):
    """Integrated GAT-CRL Agent Architecture."""

    def __init__(
        self,
        node_in_dim: int = 16,
        edge_in_dim: int = 2,
        embed_dim: int = 32,
        action_dim: int = 48,
        top_k: int = 2,
        fusion_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = GATEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=64,
            embed_dim=embed_dim,
            num_heads=4,
        )
        self.ntn = NeuralTensorNetworkSimilarity(
            embed_dim=embed_dim,
            tensor_slices=4,
        )
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.sharing = SelectivePolicySharing(
            top_k=top_k,
            fusion_weight=fusion_weight,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: Optional[torch.Tensor] = None,
        link_blockage_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, CommunicationMetrics]:
        """
        Returns:
            fused_logits      : (N, action_dim)
            values            : (N, 1)
            similarity_matrix : (N, N)
            comm_metrics      : CommunicationMetrics
        """
        # 1. GAT Node Embeddings: (N, embed_dim)
        embeddings = self.encoder(node_feats, edge_index, edge_feats)

        # 2. NTN Similarity Matrix: (N, N)
        similarity = self.ntn(embeddings)

        # 3. Local Actor and Critic predictions
        local_logits = self.actor_head(embeddings)
        values = self.critic_head(embeddings)

        # 4. Selective Policy Sharing and Soft Consensus Fusion
        fused_logits, comm_metrics = self.sharing.share_and_fuse(
            local_logits=local_logits,
            similarity_matrix=similarity,
            link_blockage_mask=link_blockage_mask,
        )

        return fused_logits, values, similarity, comm_metrics


def train_gat_crl(
    num_ues: int = 8,
    num_rbs: int = 12,
    num_steps: int = 3000,
    top_k: int = 2,
    fusion_weight: float = 0.25,
    realistic_channel: bool = False,
    lr: float = 1e-3,
    gamma: float = 0.99,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train GAT-CRL cooperative MARL policy on BaseRRMEnv."""
    print("=" * 75)
    print(f"Training GAT-CRL Cooperative MARL ({num_ues} UEs, top-k={top_k})")
    print(f"Realistic Blockage Channel : {realistic_channel}")
    print(f"Total Steps                : {num_steps:,}")
    print("=" * 75)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env_config = RRMEnvConfig(
        num_ues=num_ues,
        num_rbs=num_rbs,
        realistic_channel=realistic_channel,
    )
    env = BaseRRMEnv(config=env_config)
    env.reset(seed=seed)

    # Initial graph build to determine dimensions
    gt = GraphTopology(env)
    node_feats_np, edge_index_np, edge_feats_np = gt.to_numpy()
    node_in_dim = node_feats_np.shape[1]
    edge_in_dim = edge_feats_np.shape[1] if edge_feats_np.size > 0 else 0
    action_dim = num_rbs * 4

    agent = GATCRLAgent(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        embed_dim=32,
        action_dim=action_dim,
        top_k=top_k,
        fusion_weight=fusion_weight,
    )
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    tracker = MetricsTracker(name="gat_crl")
    step_rewards = []
    comm_bytes_history = []

    for step in range(num_steps):
        # 1. Extract graph from env state
        gt = GraphTopology(env)
        node_feats_np, edge_index_np, edge_feats_np = gt.to_numpy()

        node_t = torch.tensor(node_feats_np, dtype=torch.float32)
        edge_idx_t = torch.tensor(edge_index_np, dtype=torch.long)
        edge_feats_t = torch.tensor(edge_feats_np, dtype=torch.float32) if edge_feats_np.size > 0 else None

        # Blockage mask for realistic channel mode
        block_mask = None
        if realistic_channel and env._blockage_state is not None:
            # Mask neighbor links if both nodes are blocked
            blocked_nodes = env._blockage_state.mean(axis=1) > 0.5  # (U,)
            clear_mask = (~blocked_nodes).astype(np.float32)
            mask_mat = np.outer(clear_mask, clear_mask)
            block_mask = torch.tensor(mask_mat, dtype=torch.float32)

        # 2. Forward pass with selective sharing
        fused_logits, values, sim_matrix, comm_metrics = agent(
            node_feats=node_t,
            edge_index=edge_idx_t,
            edge_feats=edge_feats_t,
            link_blockage_mask=block_mask,
        )

        dist = torch.distributions.Categorical(logits=fused_logits)
        actions = dist.sample()  # (N,)
        log_probs = dist.log_prob(actions)

        # 3. Environment Step
        multi_action = []
        for a in actions.tolist():
            rb_idx = a // 4
            pw_idx = a % 4
            multi_action.extend([rb_idx, pw_idx])

        next_obs, r, term, trunc, step_info = env.step(np.array(multi_action, dtype=np.int64))

        # 4. Next Step Graph & Target Values
        gt_next = GraphTopology(env)
        next_nodes_np, next_edge_idx_np, next_edge_feats_np = gt_next.to_numpy()
        with torch.no_grad():
            next_node_t = torch.tensor(next_nodes_np, dtype=torch.float32)
            next_edge_idx_t = torch.tensor(next_edge_idx_np, dtype=torch.long)
            next_edge_feats_t = torch.tensor(next_edge_feats_np, dtype=torch.float32) if next_edge_feats_np.size > 0 else None
            _, next_vals, _, _ = agent(next_node_t, next_edge_idx_t, next_edge_feats_t)

        per_agent_reward = r / num_ues
        rew_tensor = torch.full_like(values, float(per_agent_reward))
        if term or trunc:
            target_vals = rew_tensor
        else:
            target_vals = rew_tensor + gamma * next_vals
        advantages = (target_vals - values).detach()

        # 5. Loss & Optimization
        actor_loss = -(log_probs.unsqueeze(-1) * advantages).mean()
        critic_loss = F.mse_loss(values, target_vals)
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_rewards.append(r)
        comm_bytes_history.append(comm_metrics.total_bytes)

        tracker.record_step(
            reward=r,
            throughput_mbps=step_info.get("mean_throughput_mbps", 0.0),
            delay_ms=step_info.get("mean_delay_ms", 0.0),
            pdr=step_info.get("pdr", 1.0),
            energy_joules=step_info.get("energy_joules", 0.0),
            sla_violated=step_info.get("sla_violations", 0) > 0,
            comm_bytes=comm_metrics.total_bytes,
        )

        if term or trunc:
            tracker.end_episode()
            env.reset()

    if tracker._current_steps:
        tracker.end_episode()

    stats = tracker.aggregate_statistics()
    early_rew = np.mean(step_rewards[:500]) if len(step_rewards) >= 500 else np.mean(step_rewards)
    late_rew = np.mean(step_rewards[-500:]) if len(step_rewards) >= 500 else np.mean(step_rewards)
    late_var = float(np.var(step_rewards[-500:])) if len(step_rewards) >= 500 else float(np.var(step_rewards))
    mean_bytes = float(np.mean(comm_bytes_history))

    print(f"\n[+] GAT-CRL Training Complete ({num_steps} steps):")
    print(f"    Early Reward (first 500 steps) : {early_rew:.4f}")
    print(f"    Late Reward  (last 500 steps)  : {late_rew:.4f}")
    print(f"    Convergence Stability Variance : {late_var:.5f} (Stabilized vs. Independent)")
    print(f"    Mean Comm Signaling Overhead   : {mean_bytes:.1f} Bytes/step (Bounded top-{top_k})")

    env.close()
    return {
        "stats": stats,
        "early_reward": early_rew,
        "late_reward": late_rew,
        "late_variance": late_var,
        "mean_comm_bytes": mean_bytes,
        "step_rewards": step_rewards,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train GAT-CRL Multi-Agent Model")
    parser.add_argument("--num-ues", type=int, default=8, help="Number of UEs / Agents")
    parser.add_argument("--num-rbs", type=int, default=12, help="Number of RBs")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k selective sharing budget")
    parser.add_argument("--realistic-channel", action="store_true", help="Enable mmWave blockage events")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    train_gat_crl(
        num_ues=args.num_ues,
        num_rbs=args.num_rbs,
        num_steps=args.steps,
        top_k=args.top_k,
        realistic_channel=args.realistic_channel,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
