# OPRA
Optimal Predictor of Resources and Activities


## Set Up
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.9+-orange.svg)

Go docs Folder
## 📖 Overview

This repository is a **fork** and extension of two foundational reinforcement learning papers. It implements a novel agent architecture that combines **search-based planning** (e.g., MCTS) with an explicit **action prediction mechanism**.

While traditional agents focus solely on maximizing reward, this version introduces an auxiliary objective: **predicting future actions**. This capability enhances the agent's understanding of environment dynamics and improves sample efficiency, particularly in complex, partial-information environments.

### 📄 Basis Papers
This project builds upon the logic and codebases of:
1.  **[Insert Title of Paper 1 - e.g., Mastering Atari with Discrete World Models (MuZero)]**
2.  **[Insert Title of Paper 2 - e.g., Predicting Future Actions of Reinforcement Learning Agents]**

---

## 🧠 Project Logic & Architecture

The core logic of this project integrates a **World Model** with a **Search Policy**, augmented by an **Action Predictor**.

### 1. The World Model (Representation & Dynamics)
The agent maintains an internal model of the environment.
*   **Representation Network**: Encodes the raw observation (e.g., pixels) into a compact latent state $s_t$.
*   **Dynamics Network**: Predicts the next latent state $s_{t+1}$ and reward $r_t$ given a state $s_t$ and an action $a_t$.

### 2. Search-Based Planning (MCTS)
Instead of acting blindly, the agent performs a **Monte Carlo Tree Search (MCTS)** within its learned latent space.
*   It simulates potential future trajectories using the Dynamics Network.
*   It balances exploration (visiting new states) and exploitation (maximizing value) to select the best immediate action.

### 3. The Action Predictor (The "Fork" Contribution)
This is the key modification from the original papers. We introduce a mechanism to predict actions explicitly:
*   **Logic**: The agent does not just output a policy $\pi$; it also outputs a predicted action distribution $\hat{a}$ for future timesteps or for other agents in the environment.
*   **Training**: The network is trained with an additional loss term that minimizes the difference between the *predicted* action and the *actual* optimal action (or the action taken by an expert/opponent).
*   **Benefit**: This forces the internal representation to capture information specifically relevant to agency and decision-making, rather than just visual reconstruction.

---

## 📂 Repository Structure

```text
├── agent/
│   ├── model.py           # Neural network architecture (Representation, Dynamics, Prediction heads)
│   ├── mcts.py            # Monte Carlo Tree Search implementation
│   └── action_pred.py     # Specific logic for the Action Prediction module
├── envs/                  # Environment wrappers (Gym, Atari, etc.)
├── training/
│   ├── buffer.py          # Replay buffer for storing trajectories
│   └── trainer.py         # Main training loop and loss calculation
├── config.py              # Hyperparameters for search and training
├── train.py               # Entry point for training the agent
├── evaluate.py            # Entry point for testing the agent
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   PyTorch (CUDA recommended for training)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/opra.git
    cd opra
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Usage

### Training the Agent
To train the agent from scratch. This will initialize the World Model and begin the MCTS collection loop.

```bash
python train.py --env CartPole-v1 --steps 100000 --predict-actions True
```

**Key Flags:**
*   `--env`: The Gym environment ID.
*   `--predict-actions`: Enables the auxiliary action prediction head.
*   `--search-depth`: Depth of the MCTS simulation.

### Evaluating Performance
To visualize the agent's performance and see the action predictions in real-time:

```bash
python evaluate.py --load-path ./checkpoints/best_model.pt --render
```

---

## 📊 Monitoring & Results

The training script logs metrics to TensorBoard. You can track:
1.  **Total Reward**: The standard RL metric.
2.  **Prediction Accuracy**: How accurately the agent predicts the actions (the specific contribution of this fork).
3.  **Search Statistics**: Average depth and value estimates.

To view logs:
```bash
tensorboard --logdir runs/
```