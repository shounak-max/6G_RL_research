"""
eval/metrics.py
===============
Shared evaluation metrics for 6G DRL/KDDL Network Optimization Testbed.

Covers:
- Network KPIs: Throughput (Mbps), Latency/Delay (ms), Packet Delivery Ratio (PDR),
  Energy Efficiency (bits/Joule), Spectral Efficiency (bps/Hz).
- RL Convergence & Policy Metrics: Cumulative reward, convergence steps,
  SLA violation rate, stability score.
- Overhead & Hardware KPIs: Inference latency (ms), communication overhead (bytes/msg).

Reference: Roadmap §4 Cross-Cutting Engineering Concerns
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class StepRecord:
    """Telemetry record for a single environment step."""
    step_idx: int
    reward: float
    throughput_mbps: float
    delay_ms: float
    pdr: float
    energy_joules: float
    sla_violated: bool
    comm_bytes: int = 0
    inference_latency_ms: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EpisodeSummary:
    """Aggregated summary of an evaluation episode."""
    episode_idx: int
    num_steps: int
    total_reward: float
    mean_reward: float
    mean_throughput_mbps: float
    mean_delay_ms: float
    mean_pdr: float
    total_energy_joules: float
    energy_efficiency_mbps_per_joule: float
    sla_violation_rate: float
    total_comm_bytes: int
    mean_inference_latency_ms: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "episode_idx": self.episode_idx,
            "num_steps": self.num_steps,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward,
            "mean_throughput_mbps": self.mean_throughput_mbps,
            "mean_delay_ms": self.mean_delay_ms,
            "mean_pdr": self.mean_pdr,
            "total_energy_joules": self.total_energy_joules,
            "energy_efficiency_mbps_per_joule": self.energy_efficiency_mbps_per_joule,
            "sla_violation_rate": self.sla_violation_rate,
            "total_comm_bytes": self.total_comm_bytes,
            "mean_inference_latency_ms": self.mean_inference_latency_ms,
        }
        d.update(self.custom_metrics)
        return d


class MetricsTracker:
    """
    Online & offline evaluation tracker for 6G RRM simulations.
    Tracks step-by-step telemetry, generates per-episode summaries,
    and aggregates across multi-episode evaluation runs.
    """

    def __init__(self, name: str = "eval_run") -> None:
        self.name = name
        self.episodes: List[EpisodeSummary] = []
        self._current_steps: List[StepRecord] = []
        self._episode_counter: int = 0

    def record_step(
        self,
        reward: float,
        throughput_mbps: float,
        delay_ms: float,
        pdr: float,
        energy_joules: float,
        sla_violated: bool,
        comm_bytes: int = 0,
        inference_latency_ms: float = 0.0,
        **custom_metrics: float,
    ) -> None:
        """Record telemetry from a single environment step."""
        step_idx = len(self._current_steps)
        self._current_steps.append(
            StepRecord(
                step_idx=step_idx,
                reward=float(reward),
                throughput_mbps=float(throughput_mbps),
                delay_ms=float(delay_ms),
                pdr=float(np.clip(pdr, 0.0, 1.0)),
                energy_joules=float(energy_joules),
                sla_violated=bool(sla_violated),
                comm_bytes=int(comm_bytes),
                inference_latency_ms=float(inference_latency_ms),
                custom_metrics=custom_metrics,
            )
        )

    def end_episode(self) -> EpisodeSummary:
        """Finalize and aggregate the current episode's steps."""
        if not self._current_steps:
            summary = EpisodeSummary(
                episode_idx=self._episode_counter,
                num_steps=0,
                total_reward=0.0,
                mean_reward=0.0,
                mean_throughput_mbps=0.0,
                mean_delay_ms=0.0,
                mean_pdr=0.0,
                total_energy_joules=0.0,
                energy_efficiency_mbps_per_joule=0.0,
                sla_violation_rate=0.0,
                total_comm_bytes=0,
                mean_inference_latency_ms=0.0,
            )
            self.episodes.append(summary)
            self._episode_counter += 1
            return summary

        num_steps = len(self._current_steps)
        total_reward = sum(s.reward for s in self._current_steps)
        mean_reward = total_reward / num_steps
        mean_tp = float(np.mean([s.throughput_mbps for s in self._current_steps]))
        mean_delay = float(np.mean([s.delay_ms for s in self._current_steps]))
        mean_pdr = float(np.mean([s.pdr for s in self._current_steps]))
        total_energy = sum(s.energy_joules for s in self._current_steps)
        total_tp = sum(s.throughput_mbps for s in self._current_steps)

        energy_eff = (total_tp / total_energy) if total_energy > 1e-6 else 0.0
        sla_violations = sum(1 for s in self._current_steps if s.sla_violated)
        sla_violation_rate = sla_violations / num_steps
        total_comm = sum(s.comm_bytes for s in self._current_steps)
        mean_lat = float(np.mean([s.inference_latency_ms for s in self._current_steps]))

        # Aggregate custom metrics
        custom_agg: Dict[str, float] = {}
        if self._current_steps[0].custom_metrics:
            keys = self._current_steps[0].custom_metrics.keys()
            for k in keys:
                vals = [s.custom_metrics.get(k, 0.0) for s in self._current_steps]
                custom_agg[f"mean_{k}"] = float(np.mean(vals))

        summary = EpisodeSummary(
            episode_idx=self._episode_counter,
            num_steps=num_steps,
            total_reward=total_reward,
            mean_reward=mean_reward,
            mean_throughput_mbps=mean_tp,
            mean_delay_ms=mean_delay,
            mean_pdr=mean_pdr,
            total_energy_joules=total_energy,
            energy_efficiency_mbps_per_joule=energy_eff,
            sla_violation_rate=sla_violation_rate,
            total_comm_bytes=total_comm,
            mean_inference_latency_ms=mean_lat,
            custom_metrics=custom_agg,
        )

        self.episodes.append(summary)
        self._episode_counter += 1
        self._current_steps.clear()
        return summary

    def aggregate_statistics(self) -> Dict[str, Any]:
        """Compute mean and 95% confidence intervals across all recorded episodes."""
        if not self.episodes:
            return {}

        metrics = [
            "total_reward",
            "mean_reward",
            "mean_throughput_mbps",
            "mean_delay_ms",
            "mean_pdr",
            "total_energy_joules",
            "energy_efficiency_mbps_per_joule",
            "sla_violation_rate",
            "total_comm_bytes",
            "mean_inference_latency_ms",
        ]

        results: Dict[str, Any] = {
            "num_episodes": len(self.episodes),
            "total_steps": sum(e.num_steps for e in self.episodes),
        }

        for m in metrics:
            vals = [getattr(e, m) for e in self.episodes]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            ci95 = 1.96 * (std_val / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            results[f"{m}_mean"] = mean_val
            results[f"{m}_std"] = std_val
            results[f"{m}_ci95"] = ci95

        return results

    def reset(self) -> None:
        """Reset all tracking buffers."""
        self.episodes.clear()
        self._current_steps.clear()
        self._episode_counter = 0


def calculate_pdr(delivered_packets: int, total_packets: int) -> float:
    """Calculate Packet Delivery Ratio safely."""
    if total_packets <= 0:
        return 1.0
    return float(np.clip(delivered_packets / total_packets, 0.0, 1.0))


def compute_spectral_efficiency(throughput_bps: float, bandwidth_hz: float) -> float:
    """Compute Spectral Efficiency in bps/Hz."""
    if bandwidth_hz <= 0:
        return 0.0
    return float(throughput_bps / bandwidth_hz)


def timer_benchmark(func, *args, **kwargs) -> Tuple[Any, float]:
    """Execute a callable and return (result, latency_in_milliseconds)."""
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    return res, latency_ms
