"""
agents/marl/gat_crl/ntn_similarity.py
=====================================
Neural Tensor Network (NTN) Pairwise Agent Similarity Scorer (Phase 2).

Computes bilinear + linear interaction scores between pairs of agent embeddings:
    s(e_i, e_j) = sigma( u^T * tanh( e_i^T * W^[1:k_t] * e_j + V * [e_i; e_j] + b ) )

The resulting pairwise similarity matrix S in [0, 1]^(N x N) determines which neighbors
each agent should selectively share policy distributions with.

Reference: Roadmap §3 Phase 2 (NTN similarity scoring)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralTensorNetworkSimilarity(nn.Module):
    """
    Neural Tensor Network (NTN) layer for measuring latent relational relevance
    between interacting 6G transceivers / UAV agents.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        tensor_slices: int = 4,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.k_slices = tensor_slices

        # Bilinear 3D Tensor: (k_slices, embed_dim, embed_dim)
        self.W_tensor = nn.Parameter(torch.empty(tensor_slices, embed_dim, embed_dim))

        # Standard Linear layer for concatenated inputs: [e_i; e_j] -> (2 * embed_dim)
        self.V_linear = nn.Linear(2 * embed_dim, tensor_slices, bias=True)

        # Output projection vector u: k_slices -> 1
        self.u_proj = nn.Linear(tensor_slices, 1, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_tensor)
        nn.init.xavier_uniform_(self.V_linear.weight)
        nn.init.zeros_(self.V_linear.bias)
        nn.init.xavier_uniform_(self.u_proj.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute full pairwise similarity matrix S for N agents.

        Parameters
        ----------
        embeddings : torch.Tensor, shape (N, embed_dim)

        Returns
        -------
        similarity_matrix : torch.Tensor, shape (N, N), values in [0, 1]
        """
        N, D = embeddings.shape
        K = self.k_slices

        # 1. Bilinear Tensor product:
        # e_i^T * W^[k] * e_j for all i, j in [0..N-1], k in [0..K-1]
        # (N, D) x (K, D, D) -> (K, N, D)
        temp = torch.einsum("nd,kdc->knc", embeddings, self.W_tensor)
        # (knc) x (md) -> (K, N, N)
        bilinear = torch.einsum("knc,mc->knm", temp, embeddings)  # shape (K, N, N)
        bilinear = bilinear.permute(1, 2, 0)                      # shape (N, N, K)

        # 2. Linear component:
        # Construct all pairwise concatenations [e_i; e_j]
        e_i_exp = embeddings.unsqueeze(1).expand(N, N, D)  # (N, N, D)
        e_j_exp = embeddings.unsqueeze(0).expand(N, N, D)  # (N, N, D)
        concat_pairs = torch.cat([e_i_exp, e_j_exp], dim=-1)  # (N, N, 2D)

        linear = self.V_linear(concat_pairs)  # (N, N, K)

        # 3. Activation and output projection
        h = torch.tanh(bilinear + linear)      # (N, N, K)
        scores = self.u_proj(h).squeeze(-1)    # (N, N)

        # 4. Normalize to [0, 1] via Sigmoid
        similarity = torch.sigmoid(scores)

        # Zero out self-loop similarity so agents don't select themselves as neighbors
        similarity = similarity * (1.0 - torch.eye(N, device=embeddings.device))

        return similarity
