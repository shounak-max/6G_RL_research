"""
envs/base_rrm_env.py
====================
Single-cell 6G Radio Resource Management (RRM) gymnasium environment.

Phase 0 — Foundations
Reference: Roadmap §3 Phase 0

State space
-----------
Per step, the observation vector contains (flattened):
  - channel_gains   : (num_ues, num_rbs)   — normalised path gain [0, 1]
  - queue_lengths   : (num_ues,)            — normalised queue backlog [0, 1]
  - ue_positions    : (num_ues, 2)          — normalised (x, y) ∈ [0, 1]²
  - sinr_estimates  : (num_ues, num_rbs)    — normalised SINR [0, 1]

Action space
------------
MultiDiscrete: for each UE, choose:
  - resource_block  : int in [0, num_rbs)
  - power_level     : int in [0, num_power_levels)

Reward
------
  r = w_tp  * normalised_throughput
    - w_dl  * normalised_delay
    - w_pw  * normalised_power
    + w_pdr * packet_delivery_ratio

The weights are set via ``SLAWeights`` and can be changed at runtime to
simulate SLA-priority shifts (used in Phase 1 cold-start tests).

Realistic-channel mode (off by default)
-----------------------------------------
Enabled via ``realistic_channel=True`` in the config.
Adds doubly-selective Rayleigh fading (time-varying multipath) and
stochastic mmWave/THz blockage events (Markov on/off per UE per RB).
Phase 0 acceptance tests run with this flag OFF.  It exists so Phase 2
can stress-test GAT-CRL against link dropouts.

Usage
-----
    from envs.base_rrm_env import BaseRRMEnv, RRMEnvConfig
    env = BaseRRMEnv()
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

CLI smoke-test
--------------
    python -m envs.base_rrm_env
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SLAWeights:
    """Reward-shaping weights mapping to physical SLA priorities."""

    throughput: float = 1.0   # weight on normalised throughput
    delay: float = 0.5        # penalty weight on normalised delay
    energy: float = 0.2       # penalty weight on normalised power
    pdr: float = 0.3          # weight on packet delivery ratio

    def as_dict(self) -> Dict[str, float]:
        return {
            "throughput": self.throughput,
            "delay": self.delay,
            "energy": self.energy,
            "pdr": self.pdr,
        }


@dataclass
class RRMEnvConfig:
    """
    Full configuration for BaseRRMEnv.

    All physical units are in SI unless noted.
    """

    # ── Topology ────────────────────────────────────────────────────────────
    num_ues: int = 8
    """Number of User Equipments (UEs)."""

    num_rbs: int = 12
    """Number of Resource Blocks (RBs) in the OFDMA frame."""

    cell_radius_m: float = 500.0
    """Cell radius in metres (UEs placed uniformly within)."""

    # ── Power ───────────────────────────────────────────────────────────────
    max_power_dbm: float = 30.0
    """Maximum transmit power in dBm (per UE)."""

    num_power_levels: int = 4
    """Discrete power levels: 0 → max_power_dbm / num_power_levels, …, n-1 → max."""

    noise_power_dbm: float = -100.0
    """Thermal noise power density in dBm."""

    # ── Traffic / Queue ─────────────────────────────────────────────────────
    max_queue_packets: int = 100
    """Maximum queue depth per UE (packets)."""

    arrival_rate_mean: float = 5.0
    """Mean packet arrival rate per step (Poisson λ)."""

    packet_size_bits: int = 1500 * 8
    """Bits per packet (default: 1500-byte Ethernet frame)."""

    # ── Channel (simple path-loss model, always active) ──────────────────
    path_loss_exponent: float = 3.5
    """Free-space path-loss exponent."""

    shadowing_std_db: float = 8.0
    """Log-normal shadowing standard deviation in dB."""

    rb_bandwidth_hz: float = 180e3
    """Bandwidth per resource block in Hz (LTE/NR standard: 180 kHz)."""

    # ── Realistic-channel mode (Phase 0 flag: OFF by default) ────────────
    realistic_channel: bool = False
    """
    When True, enables:
      - Doubly-selective Rayleigh fading (Jake's model, simplified)
      - Stochastic mmWave/THz blockage events (2-state Markov chain per UE×RB)
    Leave False for Phase 0 acceptance tests.
    """

    blockage_on_prob: float = 0.05
    """Prob. of a non-blocked link becoming blocked per step (realistic mode)."""

    blockage_off_prob: float = 0.3
    """Prob. of a blocked link recovering per step (realistic mode)."""

    doppler_max_hz: float = 100.0
    """Maximum Doppler shift in Hz for Jake's fading model."""

    # ── Episode ─────────────────────────────────────────────────────────────
    max_steps: int = 200
    """Episode length (steps)."""

    # ── SLA weights ─────────────────────────────────────────────────────────
    sla: SLAWeights = field(default_factory=SLAWeights)


# ---------------------------------------------------------------------------
# Helper: simple path-loss + shadowing channel model
# ---------------------------------------------------------------------------

def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def _dbm_to_watts(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def _compute_path_loss_db(
    distance_m: float,
    exponent: float,
    reference_distance_m: float = 1.0,
) -> float:
    """Simple log-distance path-loss model (dB)."""
    if distance_m < reference_distance_m:
        distance_m = reference_distance_m
    return 10.0 * exponent * math.log10(distance_m / reference_distance_m)


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------


class BaseRRMEnv(gym.Env):
    """
    Single-cell 6G RRM environment (Phase 0).

    See module docstring for full description.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[RRMEnvConfig] = None) -> None:
        super().__init__()
        self.cfg = config or RRMEnvConfig()

        # ── Observation space ────────────────────────────────────────────
        # Flat vector: channel_gains + sinr_estimates + queue_lengths + ue_positions
        num_ues = self.cfg.num_ues
        num_rbs = self.cfg.num_rbs
        obs_dim = (
            num_ues * num_rbs   # channel gains
            + num_ues * num_rbs  # SINR estimates
            + num_ues           # queue lengths
            + num_ues * 2       # UE positions (x, y)
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ── Action space ─────────────────────────────────────────────────
        # For each UE: choose (RB index, power level index)
        self.action_space = spaces.MultiDiscrete(
            [num_rbs, self.cfg.num_power_levels] * num_ues
        )

        # ── Internal state (initialised in reset) ────────────────────────
        self._rng: Optional[np.random.Generator] = None
        self._ue_positions: Optional[np.ndarray] = None       # (num_ues, 2)
        self._distances_m: Optional[np.ndarray] = None        # (num_ues,)
        self._channel_gains: Optional[np.ndarray] = None      # (num_ues, num_rbs)
        self._fading_phases: Optional[np.ndarray] = None      # (num_ues, num_rbs)
        self._blockage_state: Optional[np.ndarray] = None     # (num_ues, num_rbs) bool
        self._queue_lengths: Optional[np.ndarray] = None      # (num_ues,)
        self._step_count: int = 0

        # ── Cached derived quantities ─────────────────────────────────────
        self._noise_watts = _dbm_to_watts(self.cfg.noise_power_dbm)
        self._power_levels_watts = np.array(
            [
                _dbm_to_watts(
                    self.cfg.max_power_dbm - (self.cfg.num_power_levels - 1 - i)
                    * (self.cfg.max_power_dbm / self.cfg.num_power_levels)
                )
                for i in range(self.cfg.num_power_levels)
            ],
            dtype=np.float64,
        )

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        num_ues = self.cfg.num_ues
        num_rbs = self.cfg.num_rbs
        R = self.cfg.cell_radius_m

        # ── UE positions (uniform in disk) ───────────────────────────────
        r = R * np.sqrt(self._rng.uniform(0, 1, num_ues))
        theta = self._rng.uniform(0, 2 * math.pi, num_ues)
        self._ue_positions = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        self._distances_m = np.linalg.norm(self._ue_positions, axis=1).clip(min=1.0)

        # ── Channel gains (path-loss + shadowing) ────────────────────────
        self._channel_gains = self._compute_channel_gains()

        # ── Realistic-channel extras ─────────────────────────────────────
        if self.cfg.realistic_channel:
            # Fading phases for Jake's model (one per UE × RB)
            self._fading_phases = self._rng.uniform(0, 2 * math.pi, (num_ues, num_rbs))
            # Blockage state: False = clear, True = blocked
            self._blockage_state = (
                self._rng.random((num_ues, num_rbs)) < self.cfg.blockage_on_prob
            )
        else:
            self._fading_phases = None
            self._blockage_state = None

        # ── Queues (start partially filled) ──────────────────────────────
        self._queue_lengths = self._rng.integers(
            0, self.cfg.max_queue_packets // 2, size=num_ues
        ).astype(np.float64)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one RRM decision step.

        Parameters
        ----------
        action : np.ndarray
            Flat array of length 2 * num_ues.
            action[2*i]   = resource block index for UE i
            action[2*i+1] = power level index for UE i
        """
        num_ues = self.cfg.num_ues
        num_rbs = self.cfg.num_rbs

        # Parse action
        rb_assignments = np.array([action[2 * i] for i in range(num_ues)], dtype=int)
        power_indices = np.array([action[2 * i + 1] for i in range(num_ues)], dtype=int)
        tx_powers_w = self._power_levels_watts[power_indices]

        # Update channel (fading + blockage if realistic mode)
        if self.cfg.realistic_channel:
            self._update_realistic_channel()
        else:
            # Slowly vary gains with small noise (non-stationary but mild)
            self._channel_gains = (
                self._channel_gains * 0.95
                + self._compute_channel_gains() * 0.05
            )

        # Compute SINR and throughput for each UE
        sinr = self._compute_sinr(rb_assignments, tx_powers_w)
        throughput_bits = self._compute_throughput(sinr)

        # Update queues
        # 1. Capture previous queue state BEFORE adding new arrivals
        prev_queue = self._queue_lengths.copy()

        # 2. Packet arrivals (Poisson)
        arrivals = self._rng.poisson(self.cfg.arrival_rate_mean, num_ues)

        # Track the total buffer size available to deliver in this step
        queue_before_departure = np.minimum(prev_queue + arrivals, self.cfg.max_queue_packets)
        self._queue_lengths = queue_before_departure.copy()

        # 3. Departures based on throughput
        departures = np.floor(throughput_bits / self.cfg.packet_size_bits).astype(int)
        self._queue_lengths = np.maximum(self._queue_lengths - departures, 0)

        # Metrics
        packets_delivered = np.minimum(departures, queue_before_departure.astype(int))
        pdr = (
            packets_delivered.sum() / max(arrivals.sum() + prev_queue.sum(), 1)
        )
        normalised_throughput = np.clip(
            throughput_bits.sum() / (num_ues * self.cfg.rb_bandwidth_hz * 20.0), 0, 1
        )
        normalised_delay = np.clip(
            self._queue_lengths.mean() / self.cfg.max_queue_packets, 0, 1
        )
        normalised_power = np.clip(
            tx_powers_w.sum()
            / (num_ues * _dbm_to_watts(self.cfg.max_power_dbm)),
            0,
            1,
        )

        # Reward (SLA-weighted)
        w = self.cfg.sla
        reward = float(
            w.throughput * normalised_throughput
            - w.delay * normalised_delay
            - w.energy * normalised_power
            + w.pdr * pdr
        )

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.cfg.max_steps

        info = {
            "throughput_bits": float(throughput_bits.sum()),
            "pdr": float(pdr),
            "mean_queue": float(self._queue_lengths.mean()),
            "mean_sinr_db": float(10 * np.log10(sinr.mean() + 1e-12)),
            "total_power_w": float(tx_powers_w.sum()),
            "sla_violations": int((self._queue_lengths >= self.cfg.max_queue_packets).sum()),
            "step": self._step_count,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _compute_channel_gains(self) -> np.ndarray:
        """
        Compute (num_ues × num_rbs) channel gain matrix.
        Uses log-distance path-loss + log-normal shadowing.
        Returns linear absolute physical path gain values (not normalized to 1.0).
        """
        num_ues = self.cfg.num_ues
        num_rbs = self.cfg.num_rbs

        pl_db = np.array(
            [
                _compute_path_loss_db(d, self.cfg.path_loss_exponent)
                for d in self._distances_m
            ]
        )  # (num_ues,)

        # Independent shadowing per UE × RB
        shadowing_db = self._rng.normal(
            0.0, self.cfg.shadowing_std_db, (num_ues, num_rbs)
        )
        total_loss_db = pl_db[:, None] + shadowing_db  # (num_ues, num_rbs)
        gain_linear = _db_to_linear(-total_loss_db)    # (num_ues, num_rbs)

        return gain_linear.astype(np.float32)

    def _update_realistic_channel(self) -> None:
        """
        Update channel for doubly-selective fading + blockage (realistic mode).

        Fading: Jake's model approximation — rotate phase by Doppler shift each step.
        Blockage: 2-state Markov chain (on/off) per UE × RB.
        """
        num_ues = self.cfg.num_ues
        num_rbs = self.cfg.num_rbs

        # Jake's model: phase advance proportional to Doppler × time
        # We use 1 step ≈ 1 ms TTI
        phase_increment = 2 * math.pi * self.cfg.doppler_max_hz * 1e-3
        self._fading_phases = (self._fading_phases + phase_increment) % (2 * math.pi)
        fading_gain = (0.5 * (1.0 + np.cos(self._fading_phases))).astype(np.float32)

        # Blockage Markov transitions
        u = self._rng.random((num_ues, num_rbs))
        was_blocked = self._blockage_state
        newly_blocked = (~was_blocked) & (u < self.cfg.blockage_on_prob)
        newly_clear = was_blocked & (u < self.cfg.blockage_off_prob)
        self._blockage_state = (was_blocked | newly_blocked) & (~newly_clear)

        # Combine: base gain × fading × (1 if clear, 0 if blocked)
        base = self._compute_channel_gains()
        self._channel_gains = base * fading_gain * (~self._blockage_state).astype(np.float32)

    def _compute_sinr(
        self, rb_assignments: np.ndarray, tx_powers_w: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-UE SINR (linear) for the given RB/power assignment.

        Co-channel interference: UEs sharing the same RB mutually interfere.
        """
        num_ues = self.cfg.num_ues
        sinr = np.zeros(num_ues, dtype=np.float64)

        for i in range(num_ues):
            rb = rb_assignments[i]
            signal = tx_powers_w[i] * float(self._channel_gains[i, rb])
            # Interference from all other UEs on the same RB
            interference = sum(
                tx_powers_w[j] * float(self._channel_gains[j, rb])
                for j in range(num_ues)
                if j != i and rb_assignments[j] == rb
            )
            sinr[i] = signal / (interference + self._noise_watts + 1e-20)

        return sinr

    def _compute_throughput(self, sinr: np.ndarray) -> np.ndarray:
        """Shannon capacity per UE (bits per step / TTI)."""
        capacity_bps = self.cfg.rb_bandwidth_hz * np.log2(1.0 + sinr)
        # 1 ms TTI → bits per step
        bits_per_step = capacity_bps * 1e-3
        return bits_per_step

    def _get_obs(self) -> np.ndarray:
        """Flatten state components into a single [0, 1]-normalised float32 array."""
        num_ues = self.cfg.num_ues

        # Normalise positions to [-1, 1] then shift to [0, 1]
        pos_norm = (
            self._ue_positions / self.cfg.cell_radius_m * 0.5 + 0.5
        ).astype(np.float32)

        queue_norm = (self._queue_lengths / self.cfg.max_queue_packets).astype(np.float32)

        # SINR: compute a rough estimate using current channel gains and uniform power
        uniform_power = self._power_levels_watts[self.cfg.num_power_levels // 2]
        sinr_est = np.array(
            [
                (uniform_power * self._channel_gains[i, :])
                / (self._noise_watts + 1e-20)
                for i in range(num_ues)
            ],
            dtype=np.float32,
        )
        # Normalise SINR in dB scale over a realistic dynamic range (-10 dB to 40 dB)
        sinr_db = 10.0 * np.log10(sinr_est + 1e-20)
        sinr_norm = np.clip((sinr_db + 10.0) / 50.0, 0.0, 1.0).astype(np.float32)

        # Channel gains: convert absolute linear gains to dB scale and normalize between -140 dB and -40 dB
        channel_gains_db = 10.0 * np.log10(self._channel_gains + 1e-20)
        channel_gains_norm = np.clip((channel_gains_db + 140.0) / 100.0, 0.0, 1.0).astype(np.float32)

        obs = np.concatenate(
            [
                channel_gains_norm.flatten(),   # (num_ues * num_rbs,)
                sinr_norm.flatten(),             # (num_ues * num_rbs,)
                queue_norm,                       # (num_ues,)
                pos_norm.flatten(),              # (num_ues * 2,)
            ]
        )
        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "mean_queue": float(self._queue_lengths.mean()),
            "step": self._step_count,
            "realistic_channel": self.cfg.realistic_channel,
        }

    # ── Helpers for Phase 2 graph export ─────────────────────────────────────

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Return a snapshot of the current environment state for use by
        graph_topology.py and GAT-CRL agents (Phase 2).
        """
        return {
            "channel_gains": self._channel_gains.copy(),   # (num_ues, num_rbs)
            "queue_lengths": self._queue_lengths.copy(),   # (num_ues,)
            "ue_positions": self._ue_positions.copy(),     # (num_ues, 2)
            "distances_m": self._distances_m.copy(),       # (num_ues,)
            "num_ues": self.cfg.num_ues,
            "num_rbs": self.cfg.num_rbs,
            "cell_radius_m": self.cfg.cell_radius_m,
            "max_queue_packets": self.cfg.max_queue_packets,
        }

    def update_sla_weights(self, weights: SLAWeights) -> None:
        """
        Hot-swap SLA priority weights at runtime.
        Used in Phase 1 live SLA-shift tests.
        """
        self.cfg.sla = weights

    # ── Render (no-op for Phase 0) ────────────────────────────────────────────

    def render(self) -> None:  # type: ignore[override]
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("BaseRRMEnv — random-policy smoke test")
    print("=" * 60)

    cfg = RRMEnvConfig(num_ues=8, num_rbs=12, max_steps=200, realistic_channel=False)
    env = BaseRRMEnv(config=cfg)
    obs, info = env.reset(seed=42)

    print(f"Observation shape : {obs.shape}")
    print(f"Action space      : {env.action_space}")
    print(f"Observation space : {env.observation_space}")
    print("-" * 60)

    total_reward = 0.0
    total_pdr = 0.0
    sla_violations = 0
    t0 = time.perf_counter()

    for step_i in range(cfg.max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_pdr += info["pdr"]
        sla_violations += info["sla_violations"]
        if (step_i + 1) % 50 == 0:
            print(
                f"  step {step_i + 1:4d} | reward {reward:+.4f} | "
                f"PDR {info['pdr']:.3f} | mean_queue {info['mean_queue']:.1f} | "
                f"SINR {info['mean_sinr_db']:.1f} dB"
            )
        if terminated or truncated:
            break

    elapsed = time.perf_counter() - t0
    print("-" * 60)
    print(f"Total steps       : {cfg.max_steps}")
    print(f"Total reward      : {total_reward:.4f}")
    print(f"Mean PDR          : {total_pdr / cfg.max_steps:.4f}")
    print(f"Total SLA viol.   : {sla_violations}")
    print(f"Wall time         : {elapsed:.3f}s ({cfg.max_steps / elapsed:.0f} steps/s)")
    print("=" * 60)
    print("Smoke test PASSED ✓")

    env.close()
