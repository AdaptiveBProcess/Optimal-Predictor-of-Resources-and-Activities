# OPRA: Optimizing Process Resource Allocation with RL and Simulation

## Overview

OPRA (Optimizing Process Resource Allocation) is a hybrid simulation-optimization framework designed to improve business processes. At its core, it implements a **plug-and-play discrete-event process simulator** built on top of **SimPy**, driven primarily by **event logs**. It combines a data-driven Discrete Event Simulator (DES) with cutting-edge Machine Learning (ML) and Reinforcement Learning (RL) techniques to find optimal resource allocation policies.

The core idea is to:
- Take an event log
- Infer behavioral components (routing, timing, arrivals, calendars, etc.)
- Assemble them into a simulation using interchangeable **policies**
- Allow mixing paradigms: classic DES, empirical distributions, and ML-based models

While providing a powerful "what-if" analysis tool to model existing business processes, the primary objective of this project is to leverage RL agents to explore and discover new, more efficient policies for allocating resources, routing cases, and managing temporal aspects of the process. The simulator is the core of this framework, providing the environment for the RL agent to learn and improve. It's intended to be easy to use for beginners (reasonable defaults) and highly extensible for advanced users and research use cases.

## Problem Statement

In many business processes, allocating the right resource to the right task at the right time is critical for efficiency, cost reduction, and meeting service-level agreements (SLAs). However, finding the optimal allocation policy is a complex combinatorial problem, especially in dynamic environments with stochastic events and complex constraints (like resource calendars). Traditional analytical methods often fall short.

OPRA addresses this by framing the resource allocation problem as a reinforcement learning task. An RL agent learns a policy by interacting with a realistic simulation of the business process.

## Key Features

*   **Powerful Discrete Event Simulation Core:** The simulator, built on SimPy, offers a flexible and robust core for modeling complex business processes. See [High-level Architecture](#high-level-architecture) for details.
*   **Pluggable and Extensible Policies:** All behavioral aspects of the simulation (e.g., routing, processing times, resource allocation, arrivals, calendars) are defined by interchangeable policies. These can range from simple rule-based policies to empirical distributions, or advanced ML/RL-based models. Refer to [Policies (Core Extension Points)](#policies-core-extension-points) for more information.
*   **Data-Driven Initialization:** The `Initializer` automatically configures simulation parameters (e.g., arrival rates, activity durations, routing probabilities) from real-world event logs (XES/CSV format), facilitating easy setup from existing data. See [Initializer Responsibilities](#initializer-responsibilities) for more.
*   **Realistic Resource Calendars:** Model and incorporate realistic resource availability, including shifts, breaks, and holidays, influencing activity execution.
*   **Reinforcement Learning Environment:** The framework provides a standard interface (similar to OpenAI Gym) for an RL agent to interact with the simulation, receive observations, take actions, and get rewards, enabling learning of optimal policies.

## Architecture

### High-level Architecture

The simulator follows a **policy-based / hexagonal architecture**.

#### Core concepts

- **SimulatorEngine**
  - Orchestrates the simulation loop
  - Owns the SimPy environment (`env`)
  - Executes cases and activities
  - Collects an event log as output

- **SimulationSetup**
  - Immutable configuration object
  - Assembles all policies and global settings
  - Passed into the simulator at initialization

- **Initializer**
  - Builds a `SimulationSetup` from an event log
  - Encapsulates all log-to-behavior inference
  - Different initializers may exist (DES, ML-based, hybrid)

The project is structured into several key components:

*   `src/environment`: Contains the core simulation logic.
    *   `entities`: Defines the core concepts of a business process: `Case`, `Activity`, `Resource`, `Event`.
    *   `simulator`: The DES engine that drives the simulation. It manages the event queue and the simulation clock.
    *   `policies`: Defines the different policies that can be used in the simulation (e.g., for arrivals, routing, resource allocation). This is where ML/RL models can be integrated.
*   `src/agent`: Contains the implementation of the RL agent that learns the optimal policy.
*   `src/initializer`: Responsible for reading data (like event logs) and setting up the simulation environment.
*   `main.py`: The main entry point to run a simulation experiment.

### Time Semantics (Very Important)

#### Internal vs absolute time

-   **Internal simulation time**:
    -   Represented as a numeric scalar (`env.now`)
    -   Units are configurable (`seconds`, `minutes`, `hours`)
    -   All policies operate in internal time

-   **Absolute time**:
    -   Only used for:
        -   Calendar inference
        -   Arrival inference
        -   Exporting results back to real timestamps
    -   Anchored via a single `start_timestamp`

#### Design rules

-   SimPy time MUST NOT depend on Python `datetime`
-   Calendars translate absolute time → internal availability
-   Processing and waiting times are durations, not datetimes

### Policies (Core Extension Points)

All behavior is expressed via **policies**.  
Policies must be:
-   Stateless or internally self-contained
-   Replaceable without modifying the engine
-   Deterministic given a random seed (when applicable)

#### RoutingPolicy

-   Decides the next activity for a case
-   Typically inferred as a first-order Markov model:
    -   `P(next_activity | current_activity)`
-   Must explicitly support **termination** (END / None)

⚠️ Logs do NOT encode termination semantics by default.
Termination must be injected or handled via a stopping policy.

#### ProcessingTimePolicy

-   Returns the processing duration of an activity
-   Operates purely in internal time units
-   May be:
    -   Empirical (sampling from observed durations)
    -   Parametric (fitted distributions)
    -   ML-based (conditional prediction)

Zero-duration activities are allowed but dangerous.
The engine must guard against infinite zero-time loops.

#### WaitingTimePolicy (planned / optional)

-   Models delays not explained by processing time
-   Examples:
    -   Queueing delays
    -   SLA buffers
    -   Organizational latency
-   Kept separate from processing time for clarity and extensibility

#### ArrivalPolicy

-   Controls when new cases enter the system
-   Typically inferred from inter-arrival times of cases
-   Returns the next arrival time in internal units

Arrival processes are independent of routing and processing.

#### CalendarPolicy

-   Models working time availability
-   Typically inferred as a weekly (7×24) availability grid
-   Maps internal time → next working instant
-   Must not directly manipulate SimPy time

Calendars are behavioral constraints, not clocks.

#### ResourceAllocationPolicy (planned / minimal)

-   Chooses which resource executes an activity
-   May be:
    -   Random
    -   Rule-based
    -   Skill-based
    -   Learned (RL / ML)

Resource contention is handled by SimPy, not the policy.

#### Stopping / Termination Policy (important)

Because logs often contain loops, **termination must be explicit**.

Supported strategies:
-   END state injected into routing
-   Maximum number of activities per case
-   Maximum simulated time per case
-   Probabilistic stopping
-   Dedicated `StoppingPolicy`

At least one safeguard MUST exist to prevent infinite cases.

## How it Works

1.  **Initialization:** The `Initializer` reads an event log and a process model to configure the simulation parameters (e.g., statistical distributions for activity durations, arrival processes).
2.  **Simulation Loop:** The `simulator/core/engine.py` runs the simulation by processing events from an event queue in chronological order.
3.  **Decision Points:** When a decision is needed (e.g., which resource to assign to an activity), the simulator calls the corresponding policy.
4.  **Agent Interaction:** If an RL-based policy is used, the simulator passes the current state to the `agent`. The agent selects an action (e.g., chooses a resource).
5.  **State Transition:** The simulator executes the action, and the simulation state changes.
6.  **Feedback:** The simulator calculates a reward based on the outcome of the action and the new state. This reward, along with the new state, is sent back to the agent for learning.
7.  **Learning:** The agent uses this feedback to update its internal model (e.g., a neural network) to make better decisions in the future.

This cycle of `state -> action -> reward -> new state` continues, allowing the agent to learn an optimal policy for resource management within the simulated environment.

### Initializer Responsibilities

The `DESInitializer` currently performs:

-   Routing discovery (Markov transitions)
-   Processing time extraction
-   Arrival process inference
-   Calendar inference
-   Assembly of a `SimulationSetup`

Initializers are allowed to:
-   Preprocess logs
-   Inject END transitions
-   Warn about problematic patterns (loops, zero durations)

Initializers should NOT:
-   Contain simulation logic
-   Depend on SimPy

### Simulator Engine Responsibilities

The engine must:

-   Advance time ONLY via `env.timeout(x > 0)`
-   Treat routing and decisions as instantaneous
-   Respect resource contention via SimPy resources
-   Be robust to:
    -   Zero-duration activities
    -   Cyclic routing
    -   Sparse or noisy policies

The engine is NOT responsible for:
-   Learning
-   Log parsing
-   Statistical inference

### Output

-   The simulator produces an **event log**
-   The log can be exported to CSV
-   Timestamps can be:
    -   Internal time
    -   Absolute time (via `start_timestamp`)

Export logic should live close to the engine or in a dedicated exporter module.

### Design Principles (Non-negotiable)

-   Separation of concerns
-   No hard-coded behavior
-   Policies over conditionals
-   Internal time is numeric and consistent
-   Logs ≠ process semantics
-   Defaults must be safe for beginners
-   Advanced users can override everything

### Current Status

Implemented:
-   Simulator engine
-   RoutingPolicy (probabilistic)
-   EmpiricalProcessingTimePolicy
-   EmpiricalArrivalPolicy
-   WeeklyCalendarPolicy
-   DESInitializer
-   SimulationSetup

In progress / planned:
-   WaitingTimePolicy
-   ResourceAllocationPolicy
-   StoppingPolicy
-   Validation utilities
-   ML-based policies

### Explicit Non-goals (for now)

-   Full BPMN execution semantics
-   Perfect reproduction of original logs
-   UI / visualization
-   Real-time simulation

The focus is correctness, extensibility, and research value.

### Mental Model Summary

> Logs describe what happened.
> Policies describe how things behave.
> The simulator executes behavior in time.

Everything else is configuration.

## Getting Started

1.  Install the required dependencies: `pip install -r requirements.txt`
2.  Explore the `calendar_example.py` to understand how to set up and run a basic simulation.
3.  Use `main.py` as a starting point for running your own RL experiments.
