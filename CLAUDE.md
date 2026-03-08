# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup

```bash
conda create -n opra_env python=3.11.14
conda activate opra_env
pip install -r requirements.txt
# PyTorch (CPU or CUDA 12.6):
pip install torch torchvision  # CPU
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # CUDA
```

> `torch` is NOT in `requirements.txt` — it must be installed separately.

### Running

All scripts must be run from the **project root** (not from `src/`), as they use relative paths like `data/logs/...`.

```bash
# Basic DES simulation
python src/simulate.py

# RL experiment (PPO agent)
python src/main.py

# Evaluate simulated logs vs original
python src/evaluate.py
```

Output logs land in `data/simulated_logs/PurchasingExample/`.

## Architecture

OPRA is a **policy-based / hexagonal architecture** combining Discrete Event Simulation (SimPy) with Reinforcement Learning (Gymnasium + PyTorch PPO).

### Core data flow

```
Event log (CSV)
    -> Initializer.build()
    -> SimulationSetup (immutable config with all policies)
    -> SimulatorEngine (SimPy-based DES)
    -> event_log (list of dicts) -> CSV export
```

For RL:
```
SimulatorEngine (RL mode)
    -> BusinessProcessEnvironment (Gymnasium wrapper)
    -> PPOAgent selects (activity, resource) at each decision point
    -> reward = +1 per case completing within SLA threshold
```

### Key components

**`src/environment/simulator/core/engine.py` — `SimulatorEngine`**
- Two operating modes toggled by `is_rl_mode`:
  - **Standard** (`simulate()`): SimPy runs to completion, policies choose automatically.
  - **RL** (`run_until_decision()`): SimPy pauses at each routing decision, yielding control to the agent via `pending_decisions` queue and `decision_event`.
- `apply_decision(activity, resource)` resumes a paused case.
- Time advances only via `env.timeout(x > 0)` — never via Python `datetime`.

**`src/environment/simulator/core/setup.py` — `SimulationSetup`**
- Frozen dataclass holding all policies: `routing_policy`, `processing_time_policy`, `waiting_time_policy`, `arrival_policy`, `calendar_policy`, `resource_policy`.

**`src/initializer/` — Initializers**
- `DESInitializer`: builds all policies empirically from the event log (Markov routing, sampled processing times, weekly calendar grid).
- `ParametricInitializer` (extends `DESInitializer`): overrides arrival (Exponential distribution) and processing time (Normal distribution) to use fitted parametric models.
- `_build_waiting_time_policy` is a stub returning `None` in both initializers.

**`src/environment/environment.py` — `BusinessProcessEnvironment`**
- Gymnasium `Env` wrapping `SimulatorEngine`.
- `action_space`: `MultiDiscrete([num_activities, num_resources])`.
- State vector: 3 features per resource (in_use, capacity, waiting) + 1 per activity (wait count) + 1 global + 3 time features.
- Reward: `+1` per case meeting SLA, `0` otherwise.
- Activity/resource masks enforce valid transitions and skill constraints.

**`src/agent/agent.py` — `PPOAgent` / `PPOPolicy`**
- Hierarchical action selection: activity head first, then resource head conditioned on chosen activity via embedding.
- Activity and resource masks applied as `-1e9` logit fill before sampling.

### Policies

All policies live under `src/environment/simulator/policies/` (abstract base classes) with implementations in `src/environment/simulator/models/`:

| Policy | Empirical | Parametric |
|---|---|---|
| Routing | `ProbabilisticRoutingPolicy` (1st-order Markov), `SecondOrderRoutingPolicy` (2nd-order Markov) | — |
| Processing Time | `EmpiricalProcessingTimePolicy` | `NormalProcessingTimePolicy` |
| Arrival | `EmpiricalArrivalPolicy` | `ExponentialArrivalPolicy` |
| Calendar | `WeeklyCalendarPolicy` (7×24 grid) | — |
| Resource | `SkillBasedResourcePolicy` | — |

To add a new policy: implement the abstract base in `policies/`, place the implementation in `models/`, and wire it in the relevant `Initializer.build()`.

### Time semantics

- SimPy `env.now` is a numeric scalar in the configured `time_unit` (`"seconds"`, `"minutes"`, `"hours"`).
- Absolute timestamps are only used during initialization (log parsing) and when exporting results (`convert_to_absolute_time=True`), anchored by `start_timestamp`.
- **Never** pass Python `datetime` objects into the SimPy engine.

### Input data format

CSV event logs require columns (names are configurable via `LogColumnNames`):
- `caseid`, `Activity_1`, `Resource_1`, `start_timestamp`, `end_timestamp`

Simulated output logs use: `case`, `activity`, `resource`, `start`, `end`.

## Research context

This is a master's thesis project at Universidad de los Andes. Understanding the research goal is important for making correct design decisions.

### Problem framing

The process optimization problem is framed as an MDP:
- **State**: current snapshot of the running simulation (resource occupancy, activity queues, time features).
- **Action**: a joint `(activity, resource)` pair — the agent selects *both* what to do next *and* who does it. This is the key novelty over prior work, which fixes control-flow and only optimizes resource assignment.
- **Reward**: SLA compliance — a case is a success if its cycle time falls below a threshold `T` (defined as a percentile of the original log's cycle time distribution, e.g. p75 or p90).

### Masking

Activity masks use **top-k / top-p** (nucleus) filtering over learned branching probabilities to keep agent decisions within plausible process behavior. Resource masks enforce skill constraints (`resource.skills` must contain the chosen activity). Both masks are applied as `-1e9` logit fill before softmax, which is the standard approach.

### Reward function (thesis definition)

The thesis defines a two-part reward:
- **Intermediate**: `r(σ) = K / ct(σ)` — directional signal per completed case.
- **Terminal**: `+r(σ)` if `ct(σ) < T`, else `-K`.

The current implementation in `environment.py` uses a simplified version (`+1` if SLA met, `0` otherwise). The full two-part reward is the intended target.

### Planned state representation (not yet implemented)

The thesis targets a transformer-based encoder trained jointly with the DRL agent on a structured JSON representation of the simulation state (resource utilization, last N events, active cases). The current hand-crafted numeric vector is a stepping stone toward this.

### Evaluation

Simulated logs are compared against the original using the `log-distance-measures` package (`src/evaluate.py`) across:
- **Control-flow**: N-gram distance (NGD)
- **Temporal**: Absolute (AED), Circadian (CED), and Relative (RED) event distributions
- **Resource**: Circadian Workforce Distribution (CWD)
- **Congestion**: Case Arrival Rate (CAR) and Cycle Time Distribution (CTD)

### Baselines (for comparison when implementing new policies)

| Name | Activity selection | Resource selection |
|---|---|---|
| RA-RR | Proportional to branching probs | Random |
| GP-RR | Greedy (argmax branching prob) | Random |
| DM-RR | ML model (LSTM) | Random |
| DM-DRL | ML model (LSTM) | DRL agent |
| DRL-AR | DRL agent (joint) | DRL agent (joint) |

## Known issues / stubs

- `DESInitializer._build_waiting_time_policy()` returns `None` — `WaitingTimePolicy` is not yet implemented.
- `ResourceAllocationPolicy` and `StoppingPolicy` are planned but not fully implemented.
- The engine guards against infinite loops via `max_cases`, but zero-duration activities can still cause issues.
