"""
agents/marl/gat_crl/selective_sharing.py
========================================
Bandwidth-Constrained Selective Policy Distribution Sharing (Phase 2).

Key Innovations:
1. Low-Overhead Exchange: Agents exchange only discrete action probability distributions
   pi_i(a) in [0, 1]^A (e.g. 48 floats = 192 bytes), NEVER raw observations or neural network weights.
2. Top-k Sparsification: Each agent communicates exclusively with its top-k neighbors
   ranked by NTN similarity scores.
3. Bandwidth Budget Limiter: Hard-caps total transmitted bytes per step to simulate RF signaling constraints.
4. Soft Consensus Policy Fusion: Fuses neighbor policy distributions into the local actor's
   decision distribution using similarity-weighted consensus.
5. Fault-Tolerant Link Masking: Dynamically filters out blocked or high-attenuation links (Reviewer Note).

Reference: Roadmap §3 Phase 2 (Selective policy sharing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CommunicationMetrics:
    """Telemetry for signaling overhead per step."""
    total_messages: int
    total_bytes: int
    bytes_per_agent: float
    active_links_count: int


class SelectivePolicySharing(nn.Module):
    """
    Coordinates selective policy distribution sharing among decentralized MARL agents.
    """

    def __init__(
        self,
        top_k: int = 2,
        fusion_weight: float = 0.25,
        bytes_per_float: int = 4,
        bandwidth_budget_bytes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.fusion_weight = fusion_weight
        self.bytes_per_float = bytes_per_float
        self.bandwidth_budget_bytes = bandwidth_budget_bytes

    def share_and_fuse(
        self,
        local_logits: torch.Tensor,
        similarity_matrix: torch.Tensor,
        link_blockage_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, CommunicationMetrics]:
        """
        Exchange action distributions and fuse with local policy logits.

        Parameters
        ----------
        local_logits       : torch.Tensor, shape (N, action_dim)
        similarity_matrix  : torch.Tensor, shape (N, N), from NTN
        link_blockage_mask : Optional[torch.Tensor], shape (N, N), 1 if link clear, 0 if blocked

        Returns
        -------
        fused_logits : torch.Tensor, shape (N, action_dim)
        comm_metrics : CommunicationMetrics
        """
        N, A = local_logits.shape
        device = local_logits.device

        # 1. Compute local action probability distributions: (N, A)
        local_probs = F.softmax(local_logits, dim=-1)

        # 2. Mask blocked links if realistic blockage mode is active
        effective_sim = similarity_matrix.clone()
        if link_blockage_mask is not None:
            effective_sim = effective_sim * link_blockage_mask

        # 3. Top-k neighbor selection per agent
        # Clamp top_k to N - 1
        k = min(self.top_k, max(1, N - 1))
        topk_scores, topk_indices = torch.topk(effective_sim, k=k, dim=-1)  # (N, k)

        # Build sparse communication mask: (N, N)
        comm_mask = torch.zeros(N, N, device=device)
        comm_mask.scatter_(1, topk_indices, (topk_scores > 0.0).float())

        # Count active messages and bytes
        total_messages = int(comm_mask.sum().item())
        payload_bytes_per_msg = A * self.bytes_per_float
        total_bytes = total_messages * payload_bytes_per_msg

        # Apply hard bandwidth budget cap if configured
        if self.bandwidth_budget_bytes is not None and total_bytes > self.bandwidth_budget_bytes:
            # Scale down transmitted messages
            max_allowed_msgs = self.bandwidth_budget_bytes // payload_bytes_per_msg
            if max_allowed_msgs < total_messages:
                # Retain only highest scoring messages
                flat_sim = effective_sim * comm_mask
                flat_thresh = torch.topk(flat_sim.flatten(), k=max_allowed_msgs).values[-1]
                comm_mask = (flat_sim >= flat_thresh).float()
                total_messages = int(comm_mask.sum().item())
                total_bytes = total_messages * payload_bytes_per_msg

        # 4. Policy Distribution Consensus Fusion
        # Normalized neighbor weights: (N, N)
        norm_weights = (effective_sim * comm_mask)
        weight_sum = norm_weights.sum(dim=-1, keepdim=True) + 1e-8
        norm_weights = norm_weights / weight_sum  # (N, N)

        # Aggregated neighbor policy distributions: (N, N) x (N, A) -> (N, A)
        neighbor_probs = torch.matmul(norm_weights, local_probs)

        # Soft consensus: combine local logits with log neighbor distribution
        log_neighbor = torch.log(neighbor_probs + 1e-8)
        fused_logits = (1.0 - self.fusion_weight) * local_logits + self.fusion_weight * log_neighbor

        metrics = CommunicationMetrics(
            total_messages=total_messages,
            total_bytes=total_bytes,
            bytes_per_agent=total_bytes / max(1, N),
            active_links_count=total_messages,
        )

        return fused_logits, metrics
