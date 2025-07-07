# FlowZero

> **An AlphaZero-style, search-augmented reinforcement-learning agent that learns to solve _Flow Free_ puzzles from scratch.**

Flow-Zero combines a hand-rolled Monte-Carlo Tree Search (PUCT) with a lightweight ResNet policy-value network in an **Expert-Iteration** (ExIt) loop – the same recipe that powers AlphaZero.  
The project is designed to be my first exploration into reinforcement learning: most of the "interesting" algorithmic pieces are written from scratch, while heavy-lifting (tensor math, logging, CI) leans on modern libraries.

---

## ⚡️ Key Features
| Feature |
|---------|
| **Expert-Iteration loop** with PUCT-MCTS + ResNet and formal approximate policy-iteration. |
| **Custom `gymnasium` environment** for the broader RL community to use. |
| **Mostly-from-scratch core** for my own understanding |
| **Automatic curriculum & metrics** which auto-scales 5×5 all the way to 14×14; logs solve-rate, moves, tree nodes. |


---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Repository Layout](#repository-layout)
4. [TODO](#todo)
5. [Acknowledgements & References](#acknowledgements--references)

---

## Project Overview
Flow Free levels are treated as deterministic, episodic MDPs.  
Each training iteration alternates:

1. **Planning (Expert).** PUCT-MCTS runs ~800 simulations per move using the current network’s priors and value estimates.  
2. **Learning (Apprentice).** A 6-block ResNet is trained by supervised learning to imitate the tree policy (cross-entropy) and regress the final outcome (MSE).

This _Expert-Iteration_ cycle yields continual policy improvement without external supervision or human data.

---

## Quick Start

```bash
    # clone & install
    git clone https://github.com/yourusername/flowzero
    cd flowzero

    python3.12 -m venv .venv
    source .venv/Scripts/activate  # On Windows use: .venv\Scripts\activate

    pip install -r reqiurements.txt

    # run unit tests
    pytest                       # all tests should pass

    # launch self-play on the 5×5 starter pack
    python train.py --pack starter_5x5

    # benchmark against heuristic DFS
    python scripts/benchmark.py
```

## Repository Layout
```bash
flowzero/
  env/                # FlowFreeEnv + level loaders
  mcts/               # Minimal PUCT implementation
  net/                # PyTorch ResNet policy-value model
    train.py            # Expert-Iteration training loop
  scripts/
      benchmark.py    # DFS / PPO baselines
      gif_recorder.py
  tests/              # pytest suites
  examples/
      colab_demo.ipynb
configs/
  config.yaml
```
## TODO
- [ ] **GPU-batched MCTS** – integrate DeepMind `mctx` or `turbozero` for >10× speed-up.  
- [ ] **Mobile export** – distil policy to ONNX / CoreML for on-device inference.  
- [ ] **Graph encoder** – swap CNN for GNN to test large-board generalisation.  
- [ ] **Workshop paper** – draft a short report once ≥90 % solve-rate on 11×11 pack.  

Community PRs welcome 🚀

## Acknowledgements & References
| Area | Credit |
|------|--------|
| **Expert Iteration** | Anthony, Tian & Barber (2017) — “Thinking Fast and Slow with Deep Learning and Tree Search.” |
| **AlphaZero inspiration** | Silver _et al._ (2017) — “Mastering Chess and Shogi by Self-Play with a General RL Algorithm.” DeepMind. |
| **Original MCTS / PUCT** | Coulom (2006); Kocsis & Szepesvári (2006). |
| **Environment tooling** | OpenAI Gym (Brockman _et al._ 2016) → Gymnasium community fork (2023). |
| **Flow Free level data** | Independent solver repos by Rob Swindell & contributors. |

FlowZero is **not** affiliated with Big Duck Games LLC (creators of _Flow Free_) or Google DeepMind. All trademarks are the property of their respective owners.

---