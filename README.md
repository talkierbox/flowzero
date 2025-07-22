[![tests](https://github.com/talkierbox/flowzero/actions/workflows/ci.yml/badge.svg)](https://github.com/talkierbox/flowzero/actions)
# FlowZero

FlowZero is a search-augmented reinforcement-learning agent, inspired by AlphaZero, designed to solve _Flow Free_ puzzles from first principles. It combines a hand-rolled Monte Carlo Tree Search (PUCT) with a lightweight ResNet policy-value network in an Expert-Iteration loop, while relying on established libraries only for tensor operations, logging, and continuous integration.

---

## Table of Contents

- [FlowZero](#flowzero)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Repository Layout](#repository-layout)
  - [Usage](#usage)
- [Development Details](#development-details)
  - [Acknowledgments \& References](#acknowledgments--references)
  - [License \& Disclaimer](#license--disclaimer)

---

## Overview

Flow Free puzzles are cast as deterministic, episodic Markov Decision Processes (MDPs). Training proceeds in repeated Expert-Iteration cycles:

1. **Planning (Expert):**  
   PUCT-MCTS runs a fixed number of simulations per move, using the current ResNet’s policy and value estimates.  
2. **Learning (Apprentice):**  
   A 6-block ResNet is trained to  
   - imitate the tree’s move distribution (cross-entropy loss), and  
   - predict final outcomes (mean-squared error loss).  

This self-play framework yields continual policy improvement without human-labeled data.

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
│   │   ├── game.py            # Board representation, move validation, Gymnasium env
│   │   ├── generate_board.py  # Procedural puzzle generator
│   │   └── solver.py          # SAT encoder & solver for data generation
│   ├── util/                  # Helper functions and utilities
│   └── train.py               # Orchestrates Expert-Iteration training
├── tests/                     # pytest suite
│   ├── test_game/
│   ├── test_utilities/
│   └── conftest.py
├── requirements.txt
├── pyproject.toml
├── config.yaml
├── LICENSE
└── .github/
    └── workflows/ci.yml       # Linting, formatting, and test automation
```


Create and activate a Python 3.10+ virtual environment (Python 3.13 recommended):
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train or evaluate the agent:

```bash
# TODO: Train script command here
```

# Development Details

Expert-Iteration Loop
- Expert (Planning): PUCT-MCTS with a configurable number of simulations per move (default: 800).
- Apprentice (Learning): 6-block ResNet optimized via
- cross-entropy on MCTS distributions
- mean-squared error on game outcomes

TODO: Finish this

## Acknowledgments & References

Expert Iteration: Anthony, Tian & Barber (2017) [Link](https://arxiv.org/abs/1705.08439)

AlphaZero: Silver et al. (2017) [Link](https://arxiv.org/abs/1712.01815)

PUCT-MCTS: Coulom (2006); Kocsis & Szepesvári (2006)

Gymnasium: Towers et al. (2024) [Link](https://arxiv.org/abs/2407.17032)

Special Thanks: [Matt Zucker](https://github.com/mzucker), [Ben Torvaney](https://github.com/Torvaney), [Loki Chow](https://github.com/lohchness), and contributors

## License & Disclaimer
Apache 2.0 License. This project is not affiliated with Big Duck Games LLC or DeepMind.