> **⚠️ Work in Progress**: This project is under active development. Core infrastructure is complete, but the neural network training pipeline and full Expert-Iteration loop are not yet implemented.

# FlowZero

FlowZero is a search-augmented reinforcement-learning agent, inspired by AlphaZero, designed to solve _Flow Free_ puzzles from first principles. The goal is to combine a hand-rolled Monte Carlo Graph Search with a lightweight ResNet policy-value network in an Expert-Iteration loop, while relying on established libraries only for tensor operations, logging, and continuous integration.

---

## Table of Contents

- [FlowZero](#flowzero)
  - [Table of Contents](#table-of-contents)
  - [Project Status](#project-status)
  - [Overview](#overview)
  - [Repository Layout](#repository-layout)
  - [Installation](#installation)
  - [What Works Now](#what-works-now)
  - [Roadmap](#roadmap)
  - [Acknowledgments \& References](#acknowledgments--references)
  - [License \& Disclaimer](#license--disclaimer)

---

## Project Status

### ✅ Implemented
- **Flow Free game engine** with complete board representation and move validation
- **SAT-based puzzle solver** for generating verified puzzle datasets
- **Monte Carlo Graph Search (MCGS)** with UCB1 selection and configurable rollouts
- **Puzzle generation pipeline** (handcrafted, synthetic, and unsolvable examples)
- **Configuration management** via YAML
- **CI/CD pipeline** with automated testing, linting, and formatting

### 🚧 In Progress
- **MCGS refinement and testing** (currently being validated and optimized)
- **Gymnasium environment** for RL integration

### 📋 Not Yet Implemented
- **ResNet policy-value network** architecture
- **Expert-Iteration training loop** that combines MCGS with neural network learning
- **Self-play data generation** pipeline
- **Model checkpointing and evaluation** infrastructure

---

## Overview

Flow Free puzzles are cast as deterministic, episodic Markov Decision Processes (MDPs). The planned training will proceed in repeated Expert-Iteration cycles:

1. **Planning (Expert):**
   MCGS runs a fixed number of simulations per move, using the current ResNet's policy and value estimates.
2. **Learning (Apprentice):**
   A 6-block ResNet will be trained to
   - imitate the graph's move distribution (cross-entropy loss), and
   - predict final outcomes (mean-squared error loss).

This self-play framework will yield continual policy improvement without human-labeled data.

---

## Repository Layout

```plaintext
.
├── flowzero_src/
│   ├── data/
│   │   ├── handcrafted/       # Curated puzzle definitions
│   │   ├── synthetic/         # Automatically generated puzzles
│   │   └── unsolvable/        # Negative examples (e.g. unsolvable_cross.txt)
│   ├── flowfree/
│   │   ├── game.py            # ✅ Board representation, move validation
│   │   └── solver.py          # ✅ SAT encoder & solver for data generation
│   ├── gym/
│   │   └── flowfree_gym_env.py # 🚧 Gymnasium environment wrapper
│   ├── mcgs/
│   │   └── mcgs.py            # 🚧 Monte Carlo Graph Search implementation
│   ├── util/                  # ✅ Helper functions and utilities
│   ├── generate_boards.py     # ✅ Procedural puzzle generator
│   └── train.py               # 📋 Expert-Iteration training (not yet implemented)
├── tests/                     # ✅ pytest suite with comprehensive coverage
│   ├── test_game/
│   ├── test_mcgs/
│   ├── test_utilities/
│   └── conftest.py
├── requirements.txt
├── pyproject.toml
├── config.yaml
├── LICENSE
└── .github/
    └── workflows/ci.yml       # ✅ Linting, formatting, and test automation
```

---

## Installation

Create and activate a Python 3.10+ virtual environment (Python 3.13 recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run tests to verify installation:

```bash
pytest
```

---

## What Works Now

You can currently:

- **Play Flow Free puzzles** using the game engine
- **Generate puzzles** of various difficulties
- **Solve puzzles** using the SAT-based solver to verify solvability
- **Run MCGS simulations** on puzzles (experimental, under testing)
- **Use the Gymnasium environment** for custom RL experiments

---

## Roadmap

The following components are planned for future development:

1. **Complete MCGS validation** - Finish testing and refining the Monte Carlo Graph Search implementation
2. **ResNet architecture** - Implement the 6-block ResNet for policy and value prediction
3. **Training pipeline** - Build the Expert-Iteration loop combining MCGS and neural network training
4. **Self-play system** - Create infrastructure for generating training data through self-play
5. **Evaluation framework** - Develop metrics and benchmarks to track agent improvement
6. **Model management** - Add checkpointing, versioning, and model comparison tools

---

## Acknowledgments & References

Expert Iteration: Anthony, Tian & Barber (2017) [Link](https://arxiv.org/abs/1705.08439)

AlphaZero: Silver et al. (2017) [Link](https://arxiv.org/abs/1712.01815)

Gymnasium: Towers et al. (2024) [Link](https://arxiv.org/abs/2407.17032)

Special Thanks: [Matt Zucker](https://github.com/mzucker), [Ben Torvaney](https://github.com/Torvaney), [Loki Chow](https://github.com/lohchness), and contributors

## License & Disclaimer
Apache 2.0 License. This project is not affiliated with Big Duck Games LLC or DeepMind.