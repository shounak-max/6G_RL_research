"""
envs/uav_routing_env.py
=======================
DRONET-style Multi-UAV Routing Gymnasium Environment (Phase 1).

Simulates an ad-hoc 6G aerial network (Flying Ad-Hoc Network - FANET) where UAVs
relay sensor data/telemetry to a Ground Base Station (GBS) / Sink node.

Features:
- Configurable UAV density: 20 to 80 UAVs (sweepable parameter).
- 3D spatial mobility model (3D Gauss-Markov / Random Waypoint with boundary reflection).
- 3D Channel Model:
    * Air-to-Air (A2A) Friis free-space + log-normal shadowing.
    * Air-to-Ground (A2G) elevation-angle-dependent LoS/NLoS probabilistic channel.
- Energy consumption model: Propulsion power, hovering power, and RF transmit power.
- Multi-objective composite Q-Proposed reward function:
    r = w_delay * (1 / (1 + delay))
      + w_energy * normalised_residual_energy
      + w_link * link_stability_score
      + w_progress * spatial_progress_towards_destination

Reference: Roadmap §3 Phase 1 (DRONET / Q-Proposed reward)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass
class UAVSLAWeights:
    """Multi-objective reward weights for UAV routing (Q-Proposed)."""
    delay: float = 0.35       # weight on low delay / delivery latency
    energy: float = 0.25      # weight on battery preservation
    link: float = 0.20        # weight on link stability / SNR quality
    progress: float = 0.20    # weight on geometric progress to destination

    def as_dict(self) -> Dict[str, float]:
        return {
            "delay": self.delay,
            "energy": self.energy,
            "link": self.link,
            "progress": self.progress,
        }


@dataclass
class UAVEnvConfig:
    """Configuration for UAV Routing Environment."""
    num_uavs: int = 20
    """Number of UAV nodes in the network (density range: 20-80)."""

    area_width_m: float = 1000.0
    """X-dimension of operational airspace in metres."""

    area_length_m: float = 1000.0
    """Y-dimension of operational airspace in metres."""

    min_altitude_m: float = 50.0
    """Minimum UAV altitude (Z-dimension) in metres."""

    max_altitude_m: float = 200.0
    """Maximum UAV altitude (Z-dimension) in metres."""

    max_speed_mps: float = 20.0
    """Maximum horizontal/vertical flight speed in m/s."""

    comm_range_m: float = 250.0
    """Maximum effective air-to-air communication range in metres."""

    tx_power_dbm: float = 23.0
    """RF Transmit power in dBm (~200 mW)."""

    carrier_freq_ghz: float = 2.4
    """Carrier frequency in GHz."""

    noise_power_dbm: float = -95.0
    """Thermal noise floor in dBm."""

    initial_battery_joules: float = 50000.0
    """Initial battery capacity per UAV in Joules (~14 Wh)."""

    hover_power_w: float = 120.0
    """Hovering power consumption in Watts."""

    propulsion_coeff: float = 0.5
    """Velocity-dependent aerodynamic power coefficient (W / (m/s)^2)."""

    max_queue_packets: int = 50
    """Maximum packet buffer per UAV."""

    packet_arrival_rate: float = 2.0
    """Mean packet arrival rate per step (Poisson)."""

    max_steps: int = 150
    """Max episode duration in simulation steps."""

    max_candidate_neighbors: int = 5
    """Number of nearest candidate neighbors in action space."""

    sla: UAVSLAWeights = field(default_factory=UAVSLAWeights)


class UAVRoutingEnv(gym.Env):
    """
    Gymnasium environment for multi-UAV routing with Q-Proposed reward.
    Each step simulates packet forwarding, 3D mobility, channel pathloss, and battery drainage.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[UAVEnvConfig] = None) -> None:
        super().__init__()
        self.cfg = config or UAVEnvConfig()

        # Action space: Active source/relay UAV selects next hop:
        # 0: Direct transmission to Sink / GBS (if within range)
        # 1..K: Forward to K nearest neighbor UAVs
        self.k_neighbors = self.cfg.max_candidate_neighbors
        self.action_space = spaces.Discrete(self.k_neighbors + 1)

        # Observation space per active routing decision:
        # [uav_x, uav_y, uav_z, uav_battery, sink_dist, queue_occ,
        #  k_neighbor_dist_1..k, k_neighbor_snr_1..k, k_neighbor_battery_1..k]
        obs_dim = 6 + (self.k_neighbors * 3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng: Optional[np.random.Generator] = None
        self._positions: Optional[np.ndarray] = None    # (num_uavs, 3)
        self._velocities: Optional[np.ndarray] = None   # (num_uavs, 3)
        self._battery: Optional[np.ndarray] = None      # (num_uavs,)
        self._queues: Optional[np.ndarray] = None       # (num_uavs,)
        self._sink_position = np.array([
            self.cfg.area_width_m / 2.0,
            self.cfg.area_length_m / 2.0,
            0.0
        ], dtype=np.float64)

        self._active_uav_idx: int = 0
        self._step_count: int = 0
        self._total_generated_packets: int = 0
        self._total_delivered_packets: int = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng()

        num_uavs = self.cfg.num_uavs

        # 3D Positions uniformly distributed
        x = self._rng.uniform(0.0, self.cfg.area_width_m, size=num_uavs)
        y = self._rng.uniform(0.0, self.cfg.area_length_m, size=num_uavs)
        z = self._rng.uniform(self.cfg.min_altitude_m, self.cfg.max_altitude_m, size=num_uavs)
        self._positions = np.column_stack([x, y, z])

        # Velocities: Gauss-Markov initialized
        v_mag = self._rng.uniform(0.0, self.cfg.max_speed_mps, size=(num_uavs, 1))
        v_angle = self._rng.uniform(0, 2 * math.pi, size=(num_uavs, 1))
        v_pitch = self._rng.uniform(-0.2, 0.2, size=(num_uavs, 1))
        vx = v_mag * np.cos(v_angle)
        vy = v_mag * np.sin(v_angle)
        vz = v_mag * v_pitch
        self._velocities = np.column_stack([vx, vy, vz])

        self._battery = np.full(num_uavs, self.cfg.initial_battery_joules, dtype=np.float64)
        self._queues = self._rng.poisson(lam=self.cfg.packet_arrival_rate, size=num_uavs).astype(np.float64)
        self._queues = np.clip(self._queues, 0, self.cfg.max_queue_packets)

        self._active_uav_idx = int(self._rng.integers(0, num_uavs))
        self._step_count = 0
        self._total_generated_packets = int(self._queues.sum())
        self._total_delivered_packets = 0

        obs = self._get_obs(self._active_uav_idx)
        info = {
            "num_uavs": self.cfg.num_uavs,
            "pdr": 1.0,
            "mean_battery_ratio": 1.0,
            "delivered_packets": 0,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        num_uavs = self.cfg.num_uavs
        curr_idx = self._active_uav_idx
        curr_pos = self._positions[curr_idx]

        # 1. Update 3D Mobility (Gauss-Markov)
        alpha = 0.85  # Memory tuning parameter
        mean_speed = 10.0
        rand_noise = self._rng.normal(0.0, 2.0, size=self._velocities.shape)
        self._velocities = alpha * self._velocities + (1 - alpha) * mean_speed + math.sqrt(1 - alpha**2) * rand_noise
        # Limit max speed
        speed = np.linalg.norm(self._velocities, axis=1, keepdims=True)
        speed_clamped = np.clip(speed, 0, self.cfg.max_speed_mps)
        self._velocities = np.where(speed > 1e-6, self._velocities / speed * speed_clamped, self._velocities)

        # Update positions with boundary reflection
        dt = 1.0  # 1 second step
        self._positions += self._velocities * dt
        self._positions[:, 0] = np.clip(self._positions[:, 0], 0, self.cfg.area_width_m)
        self._positions[:, 1] = np.clip(self._positions[:, 1], 0, self.cfg.area_length_m)
        self._positions[:, 2] = np.clip(self._positions[:, 2], self.cfg.min_altitude_m, self.cfg.max_altitude_m)

        # 2. Update Energy Consumption
        prop_power = self.cfg.hover_power_w + self.cfg.propulsion_coeff * (speed_clamped.squeeze() ** 2)
        tx_power_w = 10.0 ** ((self.cfg.tx_power_dbm - 30.0) / 10.0)
        energy_spent = (prop_power + tx_power_w) * dt
        self._battery = np.maximum(0.0, self._battery - energy_spent)

        # 3. Process Routing Decision
        sink_dist_before = float(np.linalg.norm(curr_pos - self._sink_position))
        neighbors_idx, neighbors_dist = self._get_nearest_neighbors(curr_idx)

        packet_delivered = False
        link_stability = 0.0
        delay_score = 0.0
        progress_m = 0.0

        if action == 0:
            # Action 0: Direct transmission to Sink
            link_snr_db = self._compute_a2g_snr(curr_pos, self._sink_position)
            # Threshold for successful direct transmission
            if link_snr_db > 5.0 and sink_dist_before <= self.cfg.comm_range_m * 1.5:
                packet_delivered = True
                progress_m = sink_dist_before
                link_stability = min(1.0, max(0.0, (link_snr_db - 5.0) / 25.0))
                delay_score = 1.0  # 1 hop = minimal delay
            else:
                packet_delivered = False
                link_stability = 0.0
                delay_score = 0.1
                progress_m = 0.0
        else:
            # Action 1..K: Forward to neighbor
            neighbor_choice = action - 1
            if neighbor_choice < len(neighbors_idx):
                target_idx = neighbors_idx[neighbor_choice]
                target_pos = self._positions[target_idx]
                target_dist = neighbors_dist[neighbor_choice]

                link_snr_db = self._compute_a2a_snr(curr_pos, target_pos)
                sink_dist_after = float(np.linalg.norm(target_pos - self._sink_position))
                progress_m = sink_dist_before - sink_dist_after

                if target_dist <= self.cfg.comm_range_m and link_snr_db > 0.0:
                    link_stability = min(1.0, max(0.0, link_snr_db / 30.0))
                    delay_score = 0.7  # Multi-hop delay penalty
                    if sink_dist_after < self.cfg.comm_range_m * 0.8:
                        packet_delivered = True
                    # Buffer packet at next hop
                    self._queues[target_idx] = min(self.cfg.max_queue_packets, self._queues[target_idx] + 1)
                else:
                    link_stability = 0.0
                    delay_score = 0.0
                    progress_m = -10.0  # Dropped packet penalty
            else:
                link_stability = 0.0
                delay_score = 0.0
                progress_m = -20.0

        # Update packet counts
        if self._queues[curr_idx] > 0:
            self._queues[curr_idx] -= 1
            if packet_delivered:
                self._total_delivered_packets += 1

        # New packet arrivals
        new_arrivals = self._rng.poisson(lam=self.cfg.packet_arrival_rate)
        self._queues[curr_idx] = min(self.cfg.max_queue_packets, self._queues[curr_idx] + new_arrivals)
        self._total_generated_packets += new_arrivals

        # 4. Compute Composite Q-Proposed Reward
        # r = w_delay * (1/(1+delay)) + w_energy * residual_battery + w_link * link_stab + w_progress * norm_progress
        norm_residual_battery = float(self._battery[curr_idx] / self.cfg.initial_battery_joules)
        norm_progress = float(np.clip(progress_m / (self.cfg.comm_range_m + 1e-6), -1.0, 1.0) * 0.5 + 0.5)

        w = self.cfg.sla
        reward = float(
            w.delay * delay_score
            + w.energy * norm_residual_battery
            + w.link * link_stability
            + w.progress * norm_progress
        )

        # 5. Advance active UAV
        self._active_uav_idx = int(self._rng.integers(0, num_uavs))
        obs = self._get_obs(self._active_uav_idx)

        terminated = bool((self._battery <= 0).all())
        truncated = self._step_count >= self.cfg.max_steps

        pdr = (self._total_delivered_packets / max(1, self._total_generated_packets))
        info = {
            "pdr": float(pdr),
            "mean_battery_ratio": float(np.mean(self._battery / self.cfg.initial_battery_joules)),
            "delivered_packets": self._total_delivered_packets,
            "generated_packets": self._total_generated_packets,
            "mean_delay_ms": float(max(1.0, (1.0 - delay_score) * 50.0)),
            "energy_joules": float(energy_spent.sum()),
            "mean_throughput_mbps": float(pdr * 15.0),
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def _get_nearest_neighbors(self, uav_idx: int) -> Tuple[List[int], List[float]]:
        """Find k nearest neighboring UAVs and their distances."""
        pos = self._positions[uav_idx]
        dists = np.linalg.norm(self._positions - pos, axis=1)
        dists[uav_idx] = np.inf  # Exclude self
        nearest_indices = np.argsort(dists)[:self.k_neighbors]
        return nearest_indices.tolist(), dists[nearest_indices].tolist()

    def _compute_a2a_snr(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Air-to-Air path loss (Friis) + Log-normal shadowing."""
        dist = max(1.0, float(np.linalg.norm(pos1 - pos2)))
        c = 3e8
        wavelength = c / (self.cfg.carrier_freq_ghz * 1e9)
        fspl_db = 20 * math.log10(4 * math.pi * dist / wavelength)
        shadowing_db = float(self._rng.normal(0, 3.0)) if self._rng else 0.0
        pathloss_db = fspl_db + shadowing_db
        rx_power_dbm = self.cfg.tx_power_dbm - pathloss_db
        snr_db = rx_power_dbm - self.cfg.noise_power_dbm
        return float(snr_db)

    def _compute_a2g_snr(self, uav_pos: np.ndarray, sink_pos: np.ndarray) -> float:
        """Air-to-Ground elevation-angle probabilistic LoS channel."""
        dist = max(1.0, float(np.linalg.norm(uav_pos - sink_pos)))
        h_diff = abs(uav_pos[2] - sink_pos[2])
        elevation_deg = math.degrees(math.asin(min(1.0, h_diff / dist)))

        # ITU-R Urban environment parameters
        a, b = 9.61, 0.16
        p_los = 1.0 / (1.0 + a * math.exp(-b * (elevation_deg - a)))

        c = 3e8
        wavelength = c / (self.cfg.carrier_freq_ghz * 1e9)
        fspl_db = 20 * math.log10(4 * math.pi * dist / wavelength)
        eta_los = 1.0   # Excessive path loss LoS (dB)
        eta_nlos = 20.0 # Excessive path loss NLoS (dB)

        pl_db = fspl_db + (eta_los if self._rng.random() < p_los else eta_nlos)
        rx_power_dbm = self.cfg.tx_power_dbm - pl_db
        snr_db = rx_power_dbm - self.cfg.noise_power_dbm
        return float(snr_db)

    def _get_obs(self, uav_idx: int) -> np.ndarray:
        """Build normalized observation vector for the active UAV."""
        pos = self._positions[uav_idx]
        norm_x = pos[0] / self.cfg.area_width_m
        norm_y = pos[1] / self.cfg.area_length_m
        norm_z = (pos[2] - self.cfg.min_altitude_m) / (self.cfg.max_altitude_m - self.cfg.min_altitude_m)
        norm_batt = self._battery[uav_idx] / self.cfg.initial_battery_joules
        sink_dist = np.linalg.norm(pos - self._sink_position)
        max_dist = math.sqrt(self.cfg.area_width_m**2 + self.cfg.area_length_m**2 + self.cfg.max_altitude_m**2)
        norm_sink_dist = sink_dist / max_dist
        norm_queue = self._queues[uav_idx] / self.cfg.max_queue_packets

        base_features = [norm_x, norm_y, norm_z, norm_batt, norm_sink_dist, norm_queue]

        # Neighbor features
        neighbors_idx, neighbors_dist = self._get_nearest_neighbors(uav_idx)
        neighbor_features = []
        for i in range(self.k_neighbors):
            if i < len(neighbors_idx):
                n_idx = neighbors_idx[i]
                n_dist_norm = min(1.0, neighbors_dist[i] / self.cfg.comm_range_m)
                n_snr = max(0.0, min(1.0, self._compute_a2a_snr(pos, self._positions[n_idx]) / 35.0))
                n_batt = self._battery[n_idx] / self.cfg.initial_battery_joules
                neighbor_features.extend([n_dist_norm, n_snr, n_batt])
            else:
                neighbor_features.extend([1.0, 0.0, 0.0])

        obs = np.array(base_features + neighbor_features, dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)
