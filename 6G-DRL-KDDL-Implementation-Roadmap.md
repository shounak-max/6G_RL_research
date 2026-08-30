# 6G DRL/KDDL Network Optimization — Implementation Roadmap

**Purpose:** This is an engineering handoff document, derived from a 10-source technical
literature report on DRL/MARL/KDDL bottlenecks in 6G RRM. It translates each research gap
into a buildable simulation project with concrete modules, tasks, and acceptance criteria.

**Intended use:** Paste this whole file (or one phase at a time) into Codex / Claude Code /
another coding agent as a project spec. Each phase is scoped to be independently buildable
and testable in simulation — no live RAN hardware required for Phases 0–2.

---

## 0. Project Framing

- **Scope:** A simulated 6G RRM testbed that demonstrates, in order, (1) offline
  pre-training + cold-start mitigation, (2) decentralized cooperative MARL with
  bandwidth-efficient policy sharing, and (3) knowledge-driven model architectures
  (algorithm unrolling, hybrid model-based/data-driven pipelines) inside a closed-loop
  digital twin.
- **Non-goal:** This roadmap does not close the sim-to-real gap itself — no simulator can.
  Phase 3 builds the *interfaces* (Digital Twin sync, Lagrangian safety layer) that a real
  deployment would need, but validation against live RF hardware is out of scope for a
  coding agent working in simulation.
- **Guiding principle:** Each phase should produce a runnable benchmark with logged metrics
  before the next phase starts. Don't let the agent jump straight to Phase 3 abstractions.

---

## 1. Suggested Tech Stack

| Layer | Recommendation | Why |
|---|---|---|
| RL environments | `gymnasium` custom envs, `PettingZoo` for multi-agent | Standard, agent-friendly APIs |
| MARL training | `RLlib` (Ray) or `Stable-Baselines3` (single-agent baselines) | Mature, supports custom policies |
| Graph learning | `PyTorch Geometric` (PyG) | GAT layers, message passing for GAT-CRL and WUGNN |
| Classical ML | `scikit-learn` (Extra-Trees) | Matches the report's 96.5%-accuracy classifier |
| Simulation backbone | Custom lightweight channel/mobility simulator (start here) → optionally integrate `ns-3`/`sionna` later | Avoid over-investing in a heavy simulator before the RL loop works |
| Experiment tracking | `mlflow` or `weights & biases` | Needed across 3,000+ pretraining runs |
| Config management | `hydra` or plain YAML + `pydantic` | Sweeping SLA weight vectors, reward configs |

---

## 2. Repository Scaffold

```
6g-rrm-drl/
├── envs/
│   ├── base_rrm_env.py          # single-cell RRM Gym env (state = channel/queue/UE)
│   ├── uav_routing_env.py       # DRONET-style multi-UAV routing env
│   ├── ris_phase_env.py         # RIS meta-atom phase-shift env
│   └── graph_topology.py        # builds node/edge graph (transceivers, interference links)
├── agents/
│   ├── single_agent/            # baseline DQN / PPO agents
│   ├── marl/
│   │   ├── independent_learners.py
│   │   ├── gat_crl/
│   │   │   ├── gat_encoder.py       # multihead GAT, temporal+spatial features
│   │   │   ├── ntn_similarity.py    # Neural Tensor Network similarity scoring
│   │   │   └── selective_sharing.py # policy-distribution sharing, not raw obs/weights
│   │   └── safe_rl/
│   │       └── lagrangian_constrained.py  # primal-dual penalty for SLA/power limits
│   └── kddl/
│       ├── knowledge_assisted/  # architecture selection helpers (GNN vs LSTM)
│       ├── knowledge_fused/     # ComNet-style coarse-init + refine pipeline
│       └── knowledge_embedded/
│           └── wugnn.py         # WMMSE-unrolled GNN
├── pretraining/
│   ├── sweep_runner.py          # runs N simulations across SLA weight vectors
│   ├── expert_policy_db.py      # stores (config -> trained policy) pairs
│   └── extra_trees_selector.py  # classifier: predict best expert policy / convergence error
├── digital_twin/
│   ├── telemetry_ingest.py      # stub for real-field measurement ingestion
│   └── sync_loop.py             # calibrates sim params against (simulated) live telemetry
├── eval/
│   ├── metrics.py               # PDR, convergence steps, SLA violations, inference latency
│   └── benchmark_runner.py
├── configs/
│   └── sla_weight_vectors.yaml
└── README.md
```

Give the coding agent one directory at a time rather than the whole tree at once.

---

## 3. Phased Roadmap

### Phase 0 — Foundations (Weeks 1–3)
**Goal:** A working single-agent RRM environment and baseline agent, before any
multi-agent or knowledge-driven complexity.

- [ ] Build `base_rrm_env.py`: state = queue lengths, channel gains, UE positions; action =
      resource block / power allocation; reward = throughput − delay penalty.
- [ ] Train a baseline DQN/PPO agent on it; log convergence curve.
- [ ] Build `graph_topology.py` so any environment can export a graph representation
      (nodes = transceivers, edges = interference/adjacency) — this is reused in Phase 2.
- [ ] Set up experiment tracking (mlflow/wandb) and a fixed evaluation harness in `eval/`.

**Acceptance criteria:** Baseline agent trains to a stable reward on a small topology
(e.g., 5–10 UEs) in under an hour on a single GPU/CPU.

---

### Phase 1 — Simulation-Based Pre-Training & Cold-Start Mitigation (Months 1–4)
*Maps to Section 2 (items 1, 3) and the "Offline Pre-training" row of the gap matrix.*

**Goal:** Eliminate live cold-start risk by building an offline expert-policy database and
a classifier that selects the right policy instantly when SLA priorities shift.

- [ ] Implement `uav_routing_env.py` (DRONET-style): reward function combines packet
      delay, normalized residual energy, link stability, spatial progress (mirrors
      "Q-Proposed"). Parameterize UAV density (20–80 UAVs) as a config option.
- [ ] Implement `ris_phase_env.py` with a large discrete/continuous phase-shift action
      space; add DQN, Neural ε-greedy, and UCB baseline agents for comparison against a
      mathematically optimal (or near-optimal, brute-force-on-small-instances) baseline.
- [ ] Build `sweep_runner.py` to launch a large parallel sweep over SLA reward-weight
      vectors (report benchmark: 3,392 runs × 106 weight vectors — scale to available
      compute, but keep the *shape* of the experiment: many runs × many weight configs).
- [ ] Store each (weight vector → trained policy, convergence trace) in
      `expert_policy_db.py`.
- [ ] Train an Extra-Trees regressor/classifier (`extra_trees_selector.py`) to predict,
      from a new SLA weight vector, which expert policy to load and what convergence
      error to expect.
- [ ] Wire up a "shift SLA priority live" test: agent should load the predicted expert
      policy instead of exploring from scratch.

**Acceptance criteria (from source benchmarks — treat as targets to approach, not
guarantees at smaller scale):**
- UAV routing PDR in the 87–92% range across the tested density sweep.
- Selector accuracy approaching the 96.5% figure reported in the source (report your
  actual number — smaller sweeps will likely score lower; that's expected and worth
  noting in results).
- Measurable reduction in convergence steps when loading an expert policy vs. cold-start
  training (source claims up to ~14,000 steps saved — again, report your actual delta).

---

### Phase 2 — Cooperative Decentralized MARL: GAT-CRL (Months 4–9)
*Maps to Section 1 and the "Scalability and Coordination Dilemma" row.*

**Goal:** Move from independent learners (which suffer non-stationarity) to a
bandwidth-efficient cooperative scheme.

- [ ] Implement `independent_learners.py` as the non-stationarity baseline — deliberately
      show it degrading as agent count grows, to justify GAT-CRL.
- [ ] Implement `gat_encoder.py`: multihead Graph Attention layer over the topology graph
      from Phase 0, convolving neighbor features in temporal + spatial domains to produce
      per-agent latent representations.
- [ ] Implement `ntn_similarity.py`: Neural Tensor Network computing pairwise similarity
      scores between agents' latent representations.
- [ ] Implement `selective_sharing.py`: instead of broadcasting raw observations or full
      model weights, agents exchange **learned policy distributions** with only their
      top-k most-similar neighbors (defined by the NTN scores). Make the sharing budget
      (k, or a bandwidth cap) a tunable parameter — this is the mechanism that keeps
      signaling overhead bounded.
- [ ] Add communication-cost logging (bytes/messages exchanged per step) so you can
      directly compare GAT-CRL against full-broadcast and independent-learner baselines.

**Acceptance criteria:**
- Learning curves stabilize within roughly the source's reference window (~4,000 steps)
  under selective sharing — compare directly against the independent-learner baseline.
- Communication overhead per agent should scale sub-linearly with neighborhood size,
  not linearly with total agent count.

---

### Phase 3 — KDDL Integration, Safety, and Digital Twin (Months 9–18+)
*Maps to Sections 3 and 4, and the "Explainability & Data-Sparsity" +
"Sim-to-Real Reality Gap" rows.*

**Goal:** Replace black-box components with knowledge-structured architectures, add a
provable safety layer, and close the loop with a (simulated) digital twin.

- [ ] **Knowledge-Assisted:** Document, per environment, why a given architecture was
      chosen (GNN for graph topology, LSTM for temporal traffic) and add
      constraint-specific loss terms (primal-dual penalty terms) where relevant.
- [ ] **Knowledge-Fused (ComNet-style):** Build a two-stage pipeline in one environment
      (e.g., channel estimation or receiver design): a model-based module produces a
      coarse initialization, then a small data-driven module refines it. Compare sample
      efficiency against a pure end-to-end data-driven baseline.
- [ ] **Knowledge-Embedded (`wugnn.py`):** Implement WMMSE-unrolled GNN — take the
      classical iterative WMMSE power-allocation algorithm and unroll its iterations as
      GNN layers with learnable parameters (replacing fixed WMMSE constants). Benchmark
      inference latency vs. classical iterative WMMSE (source target: ~100x speedup) and
      scalability up to ~100 transceivers.
- [ ] **Safe/Constrained RL (`lagrangian_constrained.py`):** Implement primal-dual
      Lagrangian RL so power/rate constraints become dual penalty terms — verify that the
      agent never violates the hard constraint even during early exploration, using a
      constraint-violation-rate metric.
- [ ] **Digital Twin loop (`sync_loop.py`):** Even without real hardware, build the
      *interface*: a mock "live telemetry" stream (can be a noisier/perturbed version of
      the simulator) that periodically recalibrates simulator parameters, and a policy
      evaluation step that runs against the recalibrated twin before any action is
      "deployed." This is the scaffold a real deployment would plug into.

**Acceptance criteria:**
- WUGNN inference latency measured in milliseconds and benchmarked against classical
  WMMSE at matched transceiver counts.
- Zero (or near-zero) hard-constraint violations under the Lagrangian agent vs. a
  measurable violation rate for the unconstrained baseline.
- Digital twin sync loop demonstrably shifts the deployed policy's behavior when the
  "live telemetry" distribution is perturbed away from the base simulator.

---

## 4. Cross-Cutting Engineering Concerns (apply to every phase)

- [ ] **Reproducibility:** seed everything; log full config with every run.
- [ ] **Evaluation harness:** one shared `eval/metrics.py` (PDR, convergence steps, SLA
      violation rate, communication overhead, inference latency) so all phases/baselines
      are comparable on the same axes.
- [ ] **Ablations:** for GAT-CRL, always compare against (a) independent learners, (b)
      full-broadcast sharing, (c) no sharing. For KDDL modules, always compare against a
      pure black-box DL baseline of similar parameter count.
- [ ] **Scale honestly:** the source report's numbers (3,392 runs, 96.5% accuracy, 100x
      speedup) come from a specific setup. Treat them as *targets/reference points*, not
      guarantees — report your actual measured numbers at whatever scale you run.

---

## 5. Research Gap → Engineering Task Map (condensed reference)

| Gap | Primary sources cited | What you're building |
|---|---|---|
| Scalability vs. coordination | Huang et al. 2024; Ali et al. 2020; Althamary et al. 2019 | GAT-CRL (Phase 2) |
| Sim-to-real / cold-start / exploration risk | Nagib et al. 2023; Parvaresh & Kantarci 2023; Park et al. 2023 | Expert policy DB + Extra-Trees selector, Lagrangian RL, Digital Twin (Phases 1 & 3) |
| Explainability & data sparsity | Sun et al. 2024; Shlezinger et al. 2021; He et al. 2019 | KDDL taxonomies: assisted/fused/embedded, WUGNN (Phase 3) |
| Sustainable environmental dynamism | Nam et al. 2026; Hu et al. 2021; Dhuheir et al. 2025 | Composite reward design (Q-Proposed), meta-RL extensions (Phase 1 → 3 stretch goal) |

*Note: keep these citations in your own notes for grounding, but don't treat this
roadmap as a substitute for reading the underlying papers before implementing
algorithm-level details (e.g., the exact WMMSE unrolling scheme, exact NTN
formulation) — those need to come from the primary sources, not this summary.*

---

## 6. First Prompts to Give the Coding Agent

1. *"Build `envs/base_rrm_env.py` as described in Phase 0 of the roadmap, with a
   gymnasium-compatible interface, and a random-policy smoke test."*
2. *"Add a PPO baseline using Stable-Baselines3 against `base_rrm_env.py` and log
   reward curves to mlflow."*
3. *"Implement `graph_topology.py` that exports the environment's transceivers/UEs as a
   PyTorch Geometric graph, with interference-based edges."*

Start narrow, verify each module runs and logs metrics, then move to the next phase.
Avoid asking the agent to implement multiple phases in one pass — the report's
complexity compounds fast (graph learning + MARL + constrained RL + unrolled
networks), and each piece needs its own baseline to be meaningfully evaluated.
