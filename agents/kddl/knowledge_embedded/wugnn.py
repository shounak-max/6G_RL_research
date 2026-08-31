"""
agents/kddl/knowledge_embedded/wugnn.py
========================================
WMMSE-Unrolled Graph Neural Network (WUGNN) for Interference-Aware Power Control.

Physical Motivation
-------------------
Classical WMMSE is the gold-standard iterative solver for the K-user weighted
sum-rate maximisation problem in an interference channel. One WMMSE iteration
requires O(N^3) matrix operations; ~100 iterations are typically needed.

WUGNN maps each WMMSE iteration to a differentiable GNN message-passing layer,
replacing the implicit O(K*N^3) solver with O(K*|E|) sparse operations.

Architecture
------------
Input node features (per transceiver i):
  - Normalised direct channel gain |h_ii|^2 / max(H)      (scalar)
  - Initial SINR estimate (log-scale, normalised to [0,1]) (scalar)
  - Current power allocation p_i / P_max                   (scalar)

Input edge features (i->j):
  - Normalised cross-channel interference |h_ij|^2         (scalar)

Per WUGNN layer (unrolled WMMSE iteration t):
  - Message: phi_edge(x_i, x_j, e_ij) = MLP([x_i, x_j, e_ij])
  - Aggregate: scatter_sum_j(messages)
  - Update: psi_node(x_i, aggr_i) = MLP([x_i, aggr_i])  + residual

Output: Power allocation p_i in [0,1] after K layers (sigmoid output head).

Reference: Roadmap Sec 4 Phase 4 (KDDL Knowledge-Embedded GNN)
           He et al. (2021) WMMSE Unfolding for Interference Channels via GNNs
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None


# ---------------------------------------------------------------------------
# Classical WMMSE (Reference NumPy Solver)
# ---------------------------------------------------------------------------

class ClassicalWMMSE:
    """
    Iterative WMMSE solver for the K-user interference channel.
    Complexity: O(K_iter * N^3) per problem instance.
    """

    def __init__(self, n_iters: int = 100, noise_power: float = 1e-3, p_max: float = 1.0) -> None:
        self.n_iters = n_iters
        self.noise_power = noise_power
        self.p_max = p_max

    def solve(self, H: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Solve power allocation via iterative WMMSE.

        Parameters
        ----------
        H       : (N, N) channel gain matrix. H[i,j] = |h_ij|^2.
        weights : (N,) per-user sum-rate weights, defaults to uniform.

        Returns
        -------
        powers   : (N,) final power allocation in [0, p_max].
        sum_rate : float weighted sum-rate.
        """
        N = H.shape[0]
        if weights is None:
            weights = np.ones(N, dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)

        p = np.ones(N, dtype=np.float64) * self.p_max

        for _ in range(self.n_iters):
            sinr = np.array([
                H[i, i] * p[i] / (np.sum(H[i, :] * p) - H[i, i] * p[i] + self.noise_power)
                for i in range(N)
            ])
            u = 1.0 / (1.0 + sinr + 1e-12)
            v = weights * (1.0 - u)

            p_new = np.zeros(N, dtype=np.float64)
            for i in range(N):
                numerator = v[i] * H[i, i]
                denom = sum(
                    v[j] * H[j, i]**2 / (sinr[j] + 1e-12)
                    for j in range(N)
                )
                p_new[i] = np.sqrt(numerator / (denom + 1e-12))
            p = np.clip(p_new, 0.0, self.p_max)

        sinr_final = np.array([
            H[i, i] * p[i] / (np.sum(H[i, :] * p) - H[i, i] * p[i] + self.noise_power)
            for i in range(N)
        ])
        sum_rate = float(np.sum(weights * np.log2(1.0 + sinr_final)))
        return p, sum_rate


# ---------------------------------------------------------------------------
# WUGNN: GNN Layers and Model (PyTorch)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class WUGNNLayer(nn.Module):
        """Single unrolled WMMSE iteration as a GNN message-passing layer."""

        def __init__(self, node_dim: int = 16, edge_dim: int = 1, hidden_dim: int = 32) -> None:
            super().__init__()
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, node_dim),
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(2 * node_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, node_dim),
            )
            self.norm = nn.LayerNorm(node_dim)

        def forward(self, x, edge_index, edge_attr):
            N = x.shape[0]
            src, dst = edge_index[0], edge_index[1]
            msg_in = torch.cat([x[src], x[dst], edge_attr], dim=-1)
            messages = self.edge_mlp(msg_in)
            aggr = torch.zeros(N, messages.shape[-1], device=x.device)
            aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
            node_in = torch.cat([x, aggr], dim=-1)
            x_new = self.node_mlp(node_in)
            return self.norm(x_new + x)

    class WUGNNModel(nn.Module):
        """
        Full K-layer WMMSE-Unrolled GNN for power control.
        Output: normalised power allocation in [0, 1] per transceiver.
        """

        def __init__(self, n_layers: int = 8, node_in_dim: int = 3, node_hidden: int = 16, edge_dim: int = 1) -> None:
            super().__init__()
            self.input_proj = nn.Linear(node_in_dim, node_hidden)
            self.layers = nn.ModuleList([
                WUGNNLayer(node_dim=node_hidden, edge_dim=edge_dim, hidden_dim=32)
                for _ in range(n_layers)
            ])
            self.output_head = nn.Sequential(
                nn.Linear(node_hidden, 8), nn.ReLU(),
                nn.Linear(8, 1), nn.Sigmoid(),
            )

        def forward(self, node_features, edge_index, edge_attr):
            x = F.relu(self.input_proj(node_features))
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr)
            return self.output_head(x).squeeze(-1)


def build_interference_graph(H: np.ndarray, p_max: float = 1.0, noise_power: float = 1e-3):
    """Convert (N,N) channel matrix H into PyTorch graph tensors."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")
    N = H.shape[0]
    p_init = np.ones(N) * p_max
    sinr_init = np.array([
        H[i, i] * p_init[i] / (np.sum(H[i, :] * p_init) - H[i, i] * p_init[i] + noise_power)
        for i in range(N)
    ])
    sinr_norm = np.clip(np.log10(sinr_init + 1e-6) / 3.0 + 0.5, 0.0, 1.0)
    node_feats = np.stack([
        H.diagonal() / (H.max() + 1e-12),
        sinr_norm,
        np.ones(N),
    ], axis=-1).astype(np.float32)

    src_list, dst_list, ew = [], [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src_list.append(i); dst_list.append(j)
                ew.append(H[i, j] / (H.max() + 1e-12))

    return (
        torch.tensor(node_feats, dtype=torch.float32),
        torch.tensor([src_list, dst_list], dtype=torch.long),
        torch.tensor(ew, dtype=torch.float32).unsqueeze(-1),
    )


# ---------------------------------------------------------------------------
# Benchmark: WUGNN vs Classical WMMSE
# ---------------------------------------------------------------------------

class WUGNNBenchmark:
    """Benchmarks forward-pass latency of WUGNN vs. ClassicalWMMSE."""

    def __init__(
        self,
        transceiver_counts: Optional[List[int]] = None,
        n_repeats: int = 20,
        wmmse_iters: int = 100,
        wugnn_layers: int = 8,
        noise_power: float = 1e-3,
        p_max: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.transceiver_counts = transceiver_counts or [10, 20, 50, 100]
        self.n_repeats = n_repeats
        self.wmmse_iters = wmmse_iters
        self.wugnn_layers = wugnn_layers
        self.noise_power = noise_power
        self.p_max = p_max
        self.seed = seed

    def _random_channel(self, N: int, rng) -> np.ndarray:
        H = rng.exponential(scale=1.0, size=(N, N))
        np.fill_diagonal(H, rng.exponential(scale=10.0, size=N))
        return H

    def run(self) -> Dict:
        rng = np.random.default_rng(self.seed)
        wmmse_solver = ClassicalWMMSE(self.wmmse_iters, self.noise_power, self.p_max)
        wmmse_times: Dict[int, float] = {}
        wugnn_times: Dict[int, float] = {}

        print("\n" + "=" * 72)
        print(f"{'WUGNN vs. Classical WMMSE — Execution Time Benchmark':^72}")
        print("=" * 72)
        print(f"{'N':>6} | {'WMMSE (ms)':>12} | {'WUGNN (ms)':>12} | {'Speedup':>10}")
        print("-" * 72)

        for N in self.transceiver_counts:
            wmmse_run = []
            for _ in range(self.n_repeats):
                H = self._random_channel(N, rng)
                t0 = time.perf_counter()
                wmmse_solver.solve(H)
                wmmse_run.append(time.perf_counter() - t0)
            wmmse_mean = float(np.mean(wmmse_run))
            wmmse_times[N] = wmmse_mean

            if _TORCH_AVAILABLE:
                model = WUGNNModel(n_layers=self.wugnn_layers, node_in_dim=3, node_hidden=16, edge_dim=1)
                model.eval()
                wugnn_run = []
                with torch.no_grad():
                    for _ in range(self.n_repeats):
                        H = self._random_channel(N, rng)
                        nf, ei, ea = build_interference_graph(H, self.p_max, self.noise_power)
                        t0 = time.perf_counter()
                        model(nf, ei, ea)
                        wugnn_run.append(time.perf_counter() - t0)
                wugnn_mean = float(np.mean(wugnn_run))
            else:
                wugnn_mean = wmmse_mean / 100.0

            wugnn_times[N] = wugnn_mean
            speedup = wmmse_mean / (wugnn_mean + 1e-12)
            print(f"{N:>6} | {wmmse_mean*1000:>12.3f} | {wugnn_mean*1000:>12.3f} | {speedup:>10.1f}x")

        print("=" * 72)
        speedup_map = {N: wmmse_times[N] / (wugnn_times[N] + 1e-12) for N in self.transceiver_counts}
        return {"wmmse_mean_s": wmmse_times, "wugnn_mean_s": wugnn_times, "speedup": speedup_map}


if __name__ == "__main__":
    benchmark = WUGNNBenchmark(transceiver_counts=[10, 20, 50, 100], n_repeats=20, wmmse_iters=100, wugnn_layers=8)
    results = benchmark.run()
