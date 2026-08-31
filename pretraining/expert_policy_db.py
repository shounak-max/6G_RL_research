"""
pretraining/expert_policy_db.py
===============================
Storage, indexing, and retrieval for pre-trained expert policies.

Maps SLA weight vectors w = [w_tp, w_delay, w_energy, w_pdr] to:
- Model checkpoint path
- Empirical convergence trajectory (steps, rewards)
- State distribution statistics (mean, variance, quantiles) for distributional shift checks
- Performance KPIs (PDR, throughput, latency, SLA violation rate)

Reference: Roadmap §3 Phase 1 (Expert Policy DB)
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ExpertEntry:
    """Metadata record for a stored expert policy."""
    policy_id: str
    env_name: str
    weight_vector: List[float]  # [w_1, w_2, ..., w_d]
    checkpoint_path: str
    total_timesteps: int
    final_reward: float
    convergence_steps: int
    mean_throughput_mbps: float
    mean_delay_ms: float
    pdr: float
    sla_violation_rate: float
    state_feature_mean: List[float] = field(default_factory=list)
    state_feature_std: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checkpoint_path"] = str(Path(self.checkpoint_path).as_posix())
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExpertEntry:
        # Cross-platform path normalization
        if "checkpoint_path" in data:
            data["checkpoint_path"] = str(Path(data["checkpoint_path"].replace("\\", "/")))
        return cls(**data)


class ExpertPolicyDB:
    """
    Persistent repository of pre-trained expert policies indexed by SLA weight vectors.
    """

    def __init__(self, db_dir: str = "data/expert_policies") -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.db_dir / "index.json"
        self.entries: List[ExpertEntry] = []
        self._load_index()

    def _load_index(self) -> None:
        """Load index from JSON if present."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.entries = [ExpertEntry.from_dict(item) for item in raw]
            except Exception as e:
                print(f"[!] Failed to read index from {self.index_file}: {e}")
                self.entries = []
        else:
            self.entries = []

    def save_index(self) -> None:
        """Persist index to disk."""
        data = [e.to_dict() for e in self.entries]
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_expert(
        self,
        policy_id: str,
        env_name: str,
        weight_vector: List[float],
        model_file: str,
        total_timesteps: int,
        final_reward: float,
        convergence_steps: int,
        mean_throughput_mbps: float,
        mean_delay_ms: float,
        pdr: float,
        sla_violation_rate: float,
        state_feature_mean: Optional[List[float]] = None,
        state_feature_std: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExpertEntry:
        """Add and persist an expert policy checkpoint and metadata."""
        dest_filename = f"{policy_id}.zip"
        dest_path = self.db_dir / dest_filename
        if os.path.exists(model_file) and str(Path(model_file).resolve()) != str(dest_path.resolve()):
            shutil.copy2(model_file, dest_path)

        entry = ExpertEntry(
            policy_id=policy_id,
            env_name=env_name,
            weight_vector=[float(x) for x in weight_vector],
            checkpoint_path=str(dest_path),
            total_timesteps=total_timesteps,
            final_reward=float(final_reward),
            convergence_steps=int(convergence_steps),
            mean_throughput_mbps=float(mean_throughput_mbps),
            mean_delay_ms=float(mean_delay_ms),
            pdr=float(pdr),
            sla_violation_rate=float(sla_violation_rate),
            state_feature_mean=state_feature_mean or [],
            state_feature_std=state_feature_std or [],
            metadata=metadata or {},
        )

        # Update existing or append
        self.entries = [e for e in self.entries if e.policy_id != policy_id]
        self.entries.append(entry)
        self.save_index()
        return entry

    def find_nearest_expert(
        self,
        target_weight_vector: List[float],
        env_name: Optional[str] = None,
    ) -> Tuple[Optional[ExpertEntry], float]:
        """
        Find nearest expert by Euclidean distance in SLA weight vector space.
        Returns (best_entry, euclidean_distance).
        """
        candidates = [e for e in self.entries if env_name is None or e.env_name == env_name]
        if not candidates:
            return None, float("inf")

        target = np.array(target_weight_vector, dtype=np.float64)
        best_entry = None
        min_dist = float("inf")

        for e in candidates:
            w = np.array(e.weight_vector, dtype=np.float64)
            dist = float(np.linalg.norm(target - w))
            if dist < min_dist:
                min_dist = dist
                best_entry = e

        return best_entry, min_dist

    def get_dataset(self, env_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[ExpertEntry]]:
        """
        Export dataset of (X_weights, y_policy_indices, entries) for classifier training.
        """
        candidates = [e for e in self.entries if env_name is None or e.env_name == env_name]
        if not candidates:
            return np.empty((0, 0)), np.empty((0,)), []

        X = np.array([e.weight_vector for e in candidates], dtype=np.float64)
        y = np.arange(len(candidates), dtype=np.int64)
        return X, y, candidates

    def __len__(self) -> int:
        return len(self.entries)
