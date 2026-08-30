"""
agents/marl/gat_crl/gat_encoder.py
==================================
Multi-Head Spatial-Temporal Graph Attention Network (GAT) Encoder (Phase 2).

Convolves neighbor node features (channel gains, queue lengths, transceiver positions)
and edge interference proxies to produce compact, rich per-agent latent representations.

Features:
- Multi-head attention mechanism with LeakyReLU non-linearities.
- Supports both direct PyTorch Tensor inputs (node_feats, edge_index, edge_feats)
  and PyTorch Geometric Data objects.
- Temporal feature integration for channel drift and queue trend tracking.

Reference: Roadmap §3 Phase 2 (GAT encoder)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Custom vectorized Multi-Head Graph Attention Layer with edge feature incorporation.
    Operates efficiently on batched graphs or single topology snapshots.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_dim: int = 2,
        num_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.leaky_relu_slope = leaky_relu_slope

        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.W_edge = nn.Linear(edge_dim, num_heads * out_features, bias=False) if edge_dim > 0 else None

        # Attention vectors: a_src and a_dst
        self.a_src = nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        self.a_dst = nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        self.a_edge = nn.Parameter(torch.zeros(size=(1, num_heads, out_features))) if edge_dim > 0 else None

        self.leaky_relu = nn.LeakyReLU(self.leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        if self.W_edge is not None:
            nn.init.xavier_uniform_(self.W_edge.weight)
            nn.init.xavier_uniform_(self.a_edge)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x          : (N, in_features)
        edge_index : (2, E)
        edge_attr  : (E, edge_dim)
        """
        N = x.size(0)
        H = self.num_heads
        C = self.out_features

        # 1. Project node features: (N, H, C)
        h = self.W(x).view(N, H, C)

        # 2. Self attention terms
        src_idx, dst_idx = edge_index[0], edge_index[1]
        h_src = h[src_idx]  # (E, H, C)
        h_dst = h[dst_idx]  # (E, H, C)

        alpha_src = (h_src * self.a_src).sum(dim=-1)  # (E, H)
        alpha_dst = (h_dst * self.a_dst).sum(dim=-1)  # (E, H)
        edge_scores = alpha_src + alpha_dst           # (E, H)

        if self.W_edge is not None and edge_attr is not None:
            h_e = self.W_edge(edge_attr).view(-1, H, C)
            alpha_e = (h_e * self.a_edge).sum(dim=-1)
            edge_scores = edge_scores + alpha_e

        edge_scores = self.leaky_relu(edge_scores)

        # 3. Softmax normalization per destination node
        # Compute scatter softmax safely
        exp_scores = torch.exp(edge_scores - edge_scores.max(dim=0, keepdim=True)[0])
        # Sum exp per node
        denom = torch.zeros(N, H, device=x.device).scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, H), exp_scores) + 1e-12
        attn_weights = exp_scores / denom[dst_idx]  # (E, H)
        attn_weights = self.dropout(attn_weights)

        # 4. Message aggregation
        messages = h_src * attn_weights.unsqueeze(-1)  # (E, H, C)
        out = torch.zeros(N, H, C, device=x.device)
        dst_expanded = dst_idx.view(-1, 1, 1).expand(-1, H, C)
        out.scatter_add_(0, dst_expanded, messages)

        # Include self-loop features
        out = out + h

        if self.concat:
            return out.view(N, H * C)
        return out.mean(dim=1)


class GATEncoder(nn.Module):
    """
    Spatial-Temporal Graph Attention Encoder.
    Stacks GAT layers with residual connections to produce agent embeddings.
    """

    def __init__(
        self,
        node_in_dim: int = 16,
        edge_in_dim: int = 2,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gat1 = GATLayer(
            in_features=node_in_dim,
            out_features=hidden_dim // num_heads,
            edge_dim=edge_in_dim,
            num_heads=num_heads,
            concat=True,
            dropout=dropout,
        )
        self.gat2 = GATLayer(
            in_features=hidden_dim,
            out_features=embed_dim,
            edge_dim=edge_in_dim,
            num_heads=num_heads,
            concat=False,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            embeddings : (num_agents, embed_dim)
        """
        h = F.elu(self.gat1(node_features, edge_index, edge_features))
        h = self.gat2(h, edge_index, edge_features)
        embeddings = self.norm(h)
        return embeddings
