"""
envs/graph_topology.py
======================
Converts any BaseRRMEnv (or compatible env) into a PyTorch Geometric
``Data`` object for use by GAT-CRL and WUGNN in Phases 2 and 3.

Graph semantics
---------------
Nodes
  One node per UE (transceiver endpoint).
  Node features (per UE):
    [0]      normalised distance to BS (0–1)
    [1:1+R]  channel gains to each RB (R features)
    [1+R]    normalised queue length
    [2+R]    normalised x position
    [3+R]    normalised y position

Edges
  Undirected interference edges between pairs of UEs that share
  at least one RB assignment within a threshold interference budget.
  For Phase 0 / bootstrapping we connect all UE pairs (fully connected)
  and weight edges by the maximum cross-channel gain (proxy for
  interference potential).

  Edge features:
    [0]  max cross-channel gain (interference proxy, [0, 1])
    [1]  normalised inter-UE distance

Usage (Phase 0 — no PyG installed needed to import the module,
       only needed for ``to_pyg_data``):
-------
    from envs.base_rrm_env import BaseRRMEnv
    from envs.graph_topology import GraphTopology

    env = BaseRRMEnv()
    env.reset(seed=0)
    gt = GraphTopology(env)
    pyg_data = gt.to_pyg_data()   # requires torch-geometric

    # Or just get numpy arrays without PyG:
    node_feats, edge_index, edge_feats = gt.to_numpy()
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class GraphTopology:
    """
    Builds a graph representation from a ``BaseRRMEnv`` state snapshot.

    Parameters
    ----------
    env :
        A ``BaseRRMEnv`` (or duck-compatible) instance that has been reset.
        Internally calls ``env.get_state_dict()``.
    interference_threshold : float
        Minimum cross-channel gain (normalised, [0, 1]) to include an edge.
        Set to 0.0 to get a fully connected graph (all UE pairs).
    """

    def __init__(
        self,
        env,
        interference_threshold: float = 0.0,
    ) -> None:
        state = env.get_state_dict()
        self._channel_gains: np.ndarray = state["channel_gains"]   # (U, R)
        self._queue_lengths: np.ndarray = state["queue_lengths"]   # (U,)
        self._ue_positions: np.ndarray = state["ue_positions"]     # (U, 2)
        self._distances_m: np.ndarray = state["distances_m"]       # (U,)
        self._num_ues: int = int(state["num_ues"])
        self._num_rbs: int = int(state["num_rbs"])
        self._cell_radius: float = float(state["cell_radius_m"])
        self._max_queue_packets: float = float(state.get("max_queue_packets", 100))
        self._threshold = interference_threshold

        self._node_features, self._edge_index, self._edge_features = (
            self._build_graph()
        )

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        node_features : np.ndarray, shape (U, F_n)
        edge_index    : np.ndarray, shape (2, E)  — COO format, undirected
        edge_features : np.ndarray, shape (E, F_e)
        """
        U = self._num_ues

        # ── Node features ────────────────────────────────────────────────
        dist_norm = np.clip(self._distances_m / self._cell_radius, 0, 1)  # (U,)
        queue_norm = np.clip(
            self._queue_lengths / self._max_queue_packets, 0.0, 1.0
        )  # (U,)
        pos_norm = (
            self._ue_positions / self._cell_radius * 0.5 + 0.5
        )  # (U, 2), in [0, 1]

        # Channel gains in dB scale normalized between -140 dB and -40 dB
        channel_gains_db = 10.0 * np.log10(self._channel_gains + 1e-20)
        channel_gains_norm = np.clip((channel_gains_db + 140.0) / 100.0, 0.0, 1.0).astype(np.float32)

        node_features = np.concatenate(
            [
                dist_norm[:, None],           # (U, 1)
                channel_gains_norm,           # (U, R)
                queue_norm[:, None],          # (U, 1)
                pos_norm,                     # (U, 2)
            ],
            axis=1,
        ).astype(np.float32)  # (U, 1 + R + 1 + 2)

        # ── Edges ────────────────────────────────────────────────────────
        src_list = []
        dst_list = []
        edge_feat_list = []

        # Maximum pairwise distance for normalisation
        positions = self._ue_positions
        max_dist = max(
            np.linalg.norm(positions[i] - positions[j])
            for i in range(U)
            for j in range(i + 1, U)
        ) if U > 1 else 1.0

        for i in range(U):
            for j in range(i + 1, U):
                # Max cross-channel gain as interference proxy on normalized [0, 1] scale
                cross_gain = float(
                    (channel_gains_norm[i] * channel_gains_norm[j]).max()
                )
                if cross_gain < self._threshold:
                    continue

                dist_ij = float(np.linalg.norm(positions[i] - positions[j]))
                dist_norm_ij = dist_ij / max(max_dist, 1.0)

                edge_feat = np.array([cross_gain, dist_norm_ij], dtype=np.float32)

                # Undirected: add both directions
                src_list.extend([i, j])
                dst_list.extend([j, i])
                edge_feat_list.extend([edge_feat, edge_feat])

        if len(src_list) == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, 2), dtype=np.float32)
        else:
            edge_index = np.array([src_list, dst_list], dtype=np.int64)
            edge_features = np.stack(edge_feat_list, axis=0)

        return node_features, edge_index, edge_features

    # ── Accessors ─────────────────────────────────────────────────────────────

    def to_numpy(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the graph as raw NumPy arrays (no PyG dependency).

        Returns
        -------
        node_features : ndarray, shape (U, F_n)
        edge_index    : ndarray, shape (2, E)
        edge_features : ndarray, shape (E, F_e)
        """
        return self._node_features, self._edge_index, self._edge_features

    def to_pyg_data(self):
        """
        Return a ``torch_geometric.data.Data`` object.

        Requires ``torch`` and ``torch-geometric`` to be installed.
        Used in Phase 2 (GAT-CRL) and Phase 3 (WUGNN).
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError(
                "torch-geometric is required for to_pyg_data(). "
                "Install it following the instructions in requirements.txt."
            ) from exc

        x = torch.from_numpy(self._node_features)           # (U, F_n)
        edge_index = torch.from_numpy(self._edge_index)     # (2, E)
        edge_attr = torch.from_numpy(self._edge_features)   # (E, F_e)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        U = self._num_ues
        E = self._edge_index.shape[1] // 2  # undirected
        F_n = self._node_features.shape[1]
        return (
            f"GraphTopology(nodes={U}, edges={E}, "
            f"node_feat_dim={F_n}, edge_feat_dim=2)"
        )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from envs.base_rrm_env import BaseRRMEnv, RRMEnvConfig

    print("=" * 60)
    print("GraphTopology — smoke test")
    print("=" * 60)

    env = BaseRRMEnv(config=RRMEnvConfig(num_ues=8, num_rbs=12))
    env.reset(seed=0)

    gt = GraphTopology(env)
    node_feats, edge_index, edge_feats = gt.to_numpy()

    print(gt)
    print(f"  node_features shape : {node_feats.shape}")
    print(f"  edge_index shape    : {edge_index.shape}")
    print(f"  edge_features shape : {edge_feats.shape}")
    print()

    # Try PyG export (optional)
    try:
        data = gt.to_pyg_data()
        print(f"  PyG Data object     : {data}")
    except ImportError:
        print("  [PyG not installed — skipping to_pyg_data()]")

    print("=" * 60)
    print("GraphTopology smoke test PASSED ✓")
