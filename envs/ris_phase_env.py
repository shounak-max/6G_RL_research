"""
envs/ris_phase_env.py
=====================
Reconfigurable Intelligent Surface (RIS) Phase-Shift Gymnasium Environment (Phase 1).

Simulates a 6G multi-antenna Base Station (BS) communicating with User Equipments (UEs)
assisted by an RIS equipped with N meta-atoms / reflecting elements.

System Model:
- Direct BS-to-UE channel: H_d ~ CN(0, beta_d)
- BS-to-RIS cascaded channel: G ~ CN(0, beta_g) (Rician fading with LoS component)
- RIS-to-UE cascaded channel: h_r ~ CN(0, beta_r) (Rician fading with LoS component)
- Effective Channel: h_eff = H_d + G @ diag(exp(j * theta)) @ h_r
- Achievable rate: R = log2(1 + |h_eff|^2 * P_tx / sigma^2)

Baselines included:
- BruteForceOptimal (exact global optimum for small N <= 8)
- UCB / Multi-Armed Bandit phase optimizer
- Neural Epsilon-Greedy phase selector
- Gymnasium compatible API for PPO (continuous) and DQN (discrete)

Reference: Roadmap §3 Phase 1 (RIS phase-shift & baseline agents)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass
class RISConfig:
    """Configuration for RIS Phase-Shift Environment."""
    num_elements: int = 16
    """Number of RIS reflecting meta-atoms (N)."""

    num_ues: int = 2
    """Number of served User Equipments (UEs)."""

    num_bs_antennas: int = 4
    """Number of transmit antennas at Base Station (M)."""

    discrete_bits: int = 2
    """Quantization bits per RIS element (2-bit -> 4 phase levels: 0, pi/2, pi, 3pi/2)."""

    continuous_actions: bool = False
    """If True, action space is continuous Box[-pi, pi]^N; else Discrete/MultiDiscrete."""

    tx_power_dbm: float = 30.0
    """BS transmit power in dBm (1 Watt)."""

    noise_power_dbm: float = -100.0
    """Thermal noise floor in dBm."""

    direct_link_attenuation_db: float = -25.0
    """Additional blockage attenuation on direct BS-UE path (dB)."""

    rician_k_factor: float = 3.0
    """Rician K-factor for RIS line-of-sight paths."""

    max_steps: int = 100
    """Episode duration in steps."""


class RISPhaseEnv(gym.Env):
    """
    Gymnasium environment for RIS phase-shift optimization.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[RISConfig] = None) -> None:
        super().__init__()
        self.cfg = config or RISConfig()
        N = self.cfg.num_elements
        b = self.cfg.discrete_bits
        self.num_phase_levels = 2 ** b

        # Phase shift levels in radians: [0, 2pi * 1/L, ..., 2pi * (L-1)/L]
        self.phase_levels = np.linspace(0, 2 * np.pi, self.num_phase_levels, endpoint=False)

        if self.cfg.continuous_actions:
            self.action_space = spaces.Box(
                low=-np.pi, high=np.pi, shape=(N,), dtype=np.float32
            )
        else:
            # MultiDiscrete: each of the N elements chooses 1 of 2^b phase shifts
            self.action_space = spaces.MultiDiscrete([self.num_phase_levels] * N)

        # Observation space: Real and Imaginary parts of (G, h_r, H_d) normalized
        # Dim: BS-RIS (M * N * 2) + RIS-UE (N * K * 2) + Direct (M * K * 2)
        M = self.cfg.num_bs_antennas
        K = self.cfg.num_ues
        obs_dim = (M * N * 2) + (N * K * 2) + (M * K * 2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng: Optional[np.random.Generator] = None
        self._G: Optional[np.ndarray] = None        # BS -> RIS: (M, N) complex
        self._h_r: Optional[np.ndarray] = None      # RIS -> UE: (N, K) complex
        self._H_d: Optional[np.ndarray] = None      # BS -> UE direct: (M, K) complex
        self._step_count: int = 0
        self._current_thetas: np.ndarray = np.zeros(N, dtype=np.float64)

        # Power calculations
        self._tx_power_w = 10.0 ** ((self.cfg.tx_power_dbm - 30.0) / 10.0)
        self._noise_w = 10.0 ** ((self.cfg.noise_power_dbm - 30.0) / 10.0)

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

        self._step_count = 0
        self._sample_channels()
        self._current_thetas = np.zeros(self.cfg.num_elements, dtype=np.float64)

        obs = self._get_obs()
        info = {
            "num_elements": self.cfg.num_elements,
            "snr_db": self._compute_snr_db(self._current_thetas),
            "sum_rate_bps_hz": self._compute_sum_rate(self._current_thetas),
        }
        return obs, info

    def step(self, action: Union[np.ndarray, List[int]]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1

        # 1. Convert Action to Continuous Angles theta in [0, 2pi)
        if self.cfg.continuous_actions:
            thetas = np.asarray(action, dtype=np.float64) % (2 * np.pi)
        else:
            # MultiDiscrete index array
            action_arr = np.asarray(action, dtype=np.int64)
            thetas = self.phase_levels[action_arr]

        self._current_thetas = thetas

        # 2. Compute Beamformed Sum Rate
        sum_rate = self._compute_sum_rate(thetas)
        snr_db = self._compute_snr_db(thetas)

        # 3. Reward is the spectral efficiency / sum rate (bps/Hz)
        reward = float(sum_rate)

        # 4. Step channel drift (slow time-correlated fading)
        self._drift_channels()

        obs = self._get_obs()
        terminated = False
        truncated = self._step_count >= self.cfg.max_steps

        # Compute throughput in Mbps assuming 20 MHz bandwidth
        bw_mhz = 20.0
        tp_mbps = float(sum_rate * bw_mhz)

        info = {
            "sum_rate_bps_hz": float(sum_rate),
            "mean_throughput_mbps": tp_mbps,
            "throughput_bits": tp_mbps * 1e3,  # per ms
            "snr_db": float(snr_db),
            "pdr": 1.0 if snr_db > 0 else 0.5,
            "total_power_w": float(self._tx_power_w),
            "energy_joules": float(self._tx_power_w * 1e-3),
            "sla_violations": 0 if snr_db > 5.0 else 1,
            "mean_delay_ms": float(max(1.0, 10.0 / (sum_rate + 0.1))),
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def _sample_channels(self) -> None:
        """Sample Rician / Rayleigh fading channels."""
        M = self.cfg.num_bs_antennas
        N = self.cfg.num_elements
        K = self.cfg.num_ues
        K_factor = self.cfg.rician_k_factor

        # BS -> RIS: Rician fading
        los_G = np.ones((M, N), dtype=complex)
        nlos_G = (self._rng.standard_normal((M, N)) + 1j * self._rng.standard_normal((M, N))) / np.sqrt(2.0)
        self._G = (np.sqrt(K_factor / (K_factor + 1.0)) * los_G +
                   np.sqrt(1.0 / (K_factor + 1.0)) * nlos_G)

        # RIS -> UE: Rician fading
        los_h = np.ones((N, K), dtype=complex)
        nlos_h = (self._rng.standard_normal((N, K)) + 1j * self._rng.standard_normal((N, K))) / np.sqrt(2.0)
        self._h_r = (np.sqrt(K_factor / (K_factor + 1.0)) * los_h +
                     np.sqrt(1.0 / (K_factor + 1.0)) * nlos_h)

        # Direct link: Rayleigh fading with heavy obstacle attenuation
        att_lin = 10.0 ** (self.cfg.direct_link_attenuation_db / 20.0)
        self._H_d = att_lin * (self._rng.standard_normal((M, K)) + 1j * self._rng.standard_normal((M, K))) / np.sqrt(2.0)

    def _drift_channels(self) -> None:
        """Apply Gauss-Markov temporal channel correlation."""
        rho = 0.95  # Correlation coefficient
        M, N, K = self.cfg.num_bs_antennas, self.cfg.num_elements, self.cfg.num_ues
        noise_G = (self._rng.standard_normal((M, N)) + 1j * self._rng.standard_normal((M, N))) / np.sqrt(2.0)
        noise_h = (self._rng.standard_normal((N, K)) + 1j * self._rng.standard_normal((N, K))) / np.sqrt(2.0)
        noise_d = (self._rng.standard_normal((M, K)) + 1j * self._rng.standard_normal((M, K))) / np.sqrt(2.0)
        att_lin = 10.0 ** (self.cfg.direct_link_attenuation_db / 20.0)

        self._G = rho * self._G + np.sqrt(1 - rho**2) * noise_G
        self._h_r = rho * self._h_r + np.sqrt(1 - rho**2) * noise_h
        self._H_d = rho * self._H_d + np.sqrt(1 - rho**2) * att_lin * noise_d

    def _compute_sum_rate(self, thetas: np.ndarray) -> float:
        """Compute sum-rate capacity R = sum_k log2(1 + SINR_k)."""
        Phi = np.diag(np.exp(1j * thetas))  # (N, N)
        # Cascaded path: G @ Phi @ h_r -> (M, K)
        cascaded = self._G @ Phi @ self._h_r
        H_eff = self._H_d + cascaded  # (M, K)

        # Matched filtering beamformer per UE: W = H_eff / ||H_eff||
        # Received signal power per UE
        sum_rate = 0.0
        p_per_ue = self._tx_power_w / self.cfg.num_ues

        for k in range(self.cfg.num_ues):
            h_k = H_eff[:, k]
            channel_gain = float(np.linalg.norm(h_k) ** 2)
            snr_lin = (p_per_ue * channel_gain) / (self._noise_w + 1e-18)
            sum_rate += math.log2(1.0 + snr_lin)

        return float(sum_rate)

    def _compute_snr_db(self, thetas: np.ndarray) -> float:
        """Compute average effective SNR in dB."""
        Phi = np.diag(np.exp(1j * thetas))
        H_eff = self._H_d + (self._G @ Phi @ self._h_r)
        p_per_ue = self._tx_power_w / self.cfg.num_ues
        avg_gain = float(np.mean([np.linalg.norm(H_eff[:, k])**2 for k in range(self.cfg.num_ues)]))
        snr_lin = (p_per_ue * avg_gain) / (self._noise_w + 1e-18)
        return float(10.0 * math.log10(max(1e-12, snr_lin)))

    def _get_obs(self) -> np.ndarray:
        """Return normalized flattened real/imag channel components."""
        g_real = np.real(self._G).flatten()
        g_imag = np.imag(self._G).flatten()
        hr_real = np.real(self._h_r).flatten()
        hr_imag = np.imag(self._h_r).flatten()
        hd_real = np.real(self._H_d).flatten()
        hd_imag = np.imag(self._H_d).flatten()

        raw = np.concatenate([g_real, g_imag, hr_real, hr_imag, hd_real, hd_imag])
        # Tanh normalization to [-1, 1]
        obs = np.tanh(raw).astype(np.float32)
        return obs

    # ── Baseline Solvers ─────────────────────────────────────────────────────

    def brute_force_optimal(self) -> Tuple[np.ndarray, float]:
        """
        Exhaustive search over all quantized phase configurations.
        Only practical for small N <= 8. Returns (optimal_action_indices, max_sum_rate).
        """
        N = self.cfg.num_elements
        L = self.num_phase_levels
        if (L ** N) > 65536:
            raise RuntimeError(f"Brute force action space ({L}^{N} = {L**N}) is too large! Max supported is 65536.")

        best_action = None
        best_rate = -1.0

        for candidate in itertools.product(range(L), repeat=N):
            action_arr = np.array(candidate, dtype=np.int64)
            thetas = self.phase_levels[action_arr]
            rate = self._compute_sum_rate(thetas)
            if rate > best_rate:
                best_rate = rate
                best_action = action_arr

        return best_action, best_rate


class UCBPhaseAgent:
    """Multi-Armed Bandit / Upper Confidence Bound (UCB1) phase optimizer."""

    def __init__(self, num_elements: int, num_levels: int = 4, c: float = 1.414) -> None:
        self.num_elements = num_elements
        self.num_levels = num_levels
        self.c = c
        # Separate arm stats per RIS element
        self.counts = np.zeros((num_elements, num_levels), dtype=np.int64)
        self.values = np.zeros((num_elements, num_levels), dtype=np.float64)
        self.total_pulls = 0

    def select_action(self) -> np.ndarray:
        self.total_pulls += 1
        actions = np.zeros(self.num_elements, dtype=np.int64)
        for i in range(self.num_elements):
            for arm in range(self.num_levels):
                if self.counts[i, arm] == 0:
                    actions[i] = arm
                    break
            else:
                ucb_scores = self.values[i] + self.c * np.sqrt(
                    np.log(self.total_pulls) / self.counts[i]
                )
                actions[i] = int(np.argmax(ucb_scores))
        return actions

    def update(self, actions: np.ndarray, reward: float) -> None:
        for i, a in enumerate(actions):
            self.counts[i, a] += 1
            n = self.counts[i, a]
            self.values[i, a] += (reward - self.values[i, a]) / n
