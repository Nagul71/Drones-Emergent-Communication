# Drones – Emergent Communication

A research-focused simulation project that explores **emergent communication in multi-agent drone systems** using reinforcement learning.
The goal is to understand how autonomous drones can learn to coordinate and communicate **without predefined protocols or centralized control**.

---

## Project Overview

In traditional multi-agent systems, communication rules are manually designed.
This project investigates an alternative approach called **Emergent Communication**, where agents learn **what to communicate and how to communicate** purely through interaction and shared objectives.

The project uses a **2D drone simulation environment** where multiple drones cooperate to maximize area coverage efficiently while learning communication signals during training.

---

## Core Idea

* No hard-coded communication rules
* No central controller during execution
* Communication emerges because it improves task performance
* Agents learn through reinforcement learning

---

## Project Structure

```text
Drones-Emergent-Communication/
├── checkpoints/                # Saved trained models
├── configs/                    # Configuration files for experiments
├── learning/                   # Training logic and RL components
├── assests/                    # Images / visuals (optional)
├── coverage_history.npy        # Coverage metrics during training
├── coverage_trained.npy        # Coverage from trained agents
├── coverage_random.npy         # Coverage from random baseline
├── reward_history.npy          # Reward values during training
├── plot_metrics.py             # Plot rewards and coverage graphs
├── run_simulation.py           # Main simulation entry point
└── README.md                   # Project documentation
```
---

## Environment Description

* 2D continuous space
* Finite boundaries
* Multiple autonomous drones

Each drone has:

* Limited sensing
* Battery constraints
* No global state access

---

## Task Definition

### Area Coverage Task

* Drones must collectively explore and cover a region
* Overlapping coverage is penalized
* Efficient coverage and energy usage are rewarded
* Requires coordination and cooperation

---

##  Learning Framework

* Multi-Agent Reinforcement Learning
* Shared reward for cooperative behavior
* Centralized training, decentralized execution
* Communication is learned implicitly

---

##  Metrics & Evaluation

The system evaluates performance using:

* Area coverage percentage
* Energy efficiency
* Training reward trends
* Comparison with random baseline agents

Metrics are stored as `.npy` files and visualized using `plot_metrics.py`.

---

## How to Run

### Clone the repository

git clone [https://github.com/Nagul71/Drones-Emergent-Communication.git](https://github.com/Nagul71/Drones-Emergent-Communication.git)
cd Drones-Emergent-Communication

### Run simulation / training

python run_simulation.py

### Plot results

python plot_metrics.py

---

## Requirements

* Python 3.7+
* NumPy
* Matplotlib
* PyTorch

Install dependencies:

pip install numpy matplotlib torch

---

## Project Scope

### What this project IS

* A system-level AI project
* Study of decentralized coordination
* Practical emergent communication simulation

### What this project IS NOT

* Real-world drone deployment
* Hardware-based system
* New RL algorithm proposal

---

## Applications

* Drone swarms
* Search and rescue systems
* Autonomous vehicle coordination
* Warehouse robotics
* Distributed monitoring systems

