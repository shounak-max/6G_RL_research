# 6G DRL/KDDL Network Optimization Research Testbed

A simulated 6G Radio Resource Management (RRM) testbed demonstrating:
1. **Phase 0** — Single-agent baseline (DQN/PPO) with a gymnasium RRM environment
2. **Phase 1** — Offline pre-training & cold-start mitigation (SLA sweep, expert policy DB, Extra-Trees selector)
3. **Phase 2** — Cooperative decentralized MARL via GAT-CRL (Graph Attention + selective policy sharing)
4. **Phase 3** — Knowledge-Driven Deep Learning (KDDL): WMMSE-unrolled GNN, Lagrangian safety layer, Digital Twin sync loop

---

## Quick Start

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Install PyG sparse deps matching your torch/CUDA version:
# CPU example:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

### 2. Smoke-test the environment

```bash
python -m envs.base_rrm_env
```

Expected output: random-policy rollout for 200 steps, printed step metrics.

### 3. Train a baseline agent (PPO)

```bash
python agents/single_agent/train.py \
  --algo ppo \
  --total-timesteps 500000 \
  --run-name baseline_ppo \
  --sla-profile balanced
```

### 4. View experiment results

```bash
mlflow ui
# Open http://127.0.0.1:5000
```

### 5. Run the evaluation benchmark

```bash
python eval/benchmark_runner.py \
  --model-path runs/baseline_ppo/model.zip \
  --num-episodes 100
```

---

## Repository Structure

```
6G_RL_research/
├── envs/
│   ├── base_rrm_env.py          # Phase 0: single-cell RRM Gym env
│   ├── graph_topology.py        # Phase 0: PyG graph export (reused in Phase 2)
│   ├── uav_routing_env.py       # Phase 1: DRONET-style UAV routing env (stub)
│   └── ris_phase_env.py         # Phase 1: RIS phase-shift env (stub)
├── agents/
│   ├── single_agent/
│   │   └── train.py             # Phase 0: PPO/DQN training loop
│   ├── marl/
│   │   ├── independent_learners.py   # Phase 2: non-stationarity baseline
│   │   ├── gat_crl/                  # Phase 2: GAT-CRL cooperative MARL
│   │   └── safe_rl/                  # Phase 3: Lagrangian constrained RL
│   └── kddl/
│       ├── knowledge_assisted/       # Phase 3
│       ├── knowledge_fused/          # Phase 3
│       └── knowledge_embedded/
│           └── wugnn.py              # Phase 3: WMMSE-unrolled GNN
├── pretraining/
│   ├── sweep_runner.py          # Phase 1: parallel SLA-weight sweep
│   ├── expert_policy_db.py      # Phase 1: (weight_vec → policy) store
│   └── extra_trees_selector.py  # Phase 1: cold-start policy selector
├── digital_twin/
│   ├── telemetry_ingest.py      # Phase 3: live telemetry stub
│   └── sync_loop.py             # Phase 3: env recalibration loop
├── eval/
│   ├── metrics.py               # Shared metrics (all phases)
│   └── benchmark_runner.py      # Standardized evaluation harness
├── configs/
│   └── sla_weight_vectors.yaml  # SLA weight profiles for sweep & training
├── requirements.txt
└── README.md
```

---

## Phase Status

| Phase | Status | Description |
|---|---|---|
| **0 — Foundations** | ✅ Active | BaseRRMEnv, PPO/DQN, graph topology, eval harness |
| **1 — Pre-training** | 🔲 Stub | UAV env, RIS env, SLA sweep, Extra-Trees selector |
| **2 — GAT-CRL MARL** | 🔲 Stub | Cooperative MARL with selective policy sharing |
| **3 — KDDL + Safety** | 🔲 Stub | WUGNN, Lagrangian RL, Digital Twin |

---

## Research Gap → Engineering Task Map

| Gap | Building |
|---|---|
| Scalability vs. coordination | GAT-CRL (Phase 2) |
| Sim-to-real / cold-start | Expert policy DB + Extra-Trees + Lagrangian RL (Phases 1 & 3) |
| Explainability & data sparsity | KDDL: assisted/fused/embedded, WUGNN (Phase 3) |
| Sustainable environmental dynamism | Composite reward (Q-Proposed), Digital Twin (Phase 3) |

---

## Reproducibility

All runs seed NumPy, PyTorch, and the gymnasium environment via a single `--seed` flag.
Full configs are logged to mlflow with every experiment run.
