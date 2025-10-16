# RiverSwim & SixArms Environments

Minimal implementations of the RiverSwim and SixArms environments described in:

> [**Alexander L. Strehl, Michael L. Littman,**
> *An analysis of model-based Interval Estimation for Markov Decision Processes,
Journal of Computer and System Sciences,
Volume 74, Issue 8
(2008).* ](https://www.sciencedirect.com/science/article/pii/S0022000008000767)

## Overview

The paper introduces two small, structured Markov Decision Processes that stress-test exploration algorithms:

- **RiverSwim:** A 1D chain with a strong current. Swimming right is risky but yields a very large reward in the rightmost state, while the leftmost state gives a small but reliable reward.
- **SixArms:** A hub-and-spoke layout inspired by a stochastic multi-armed bandit. The hub state provides six different arms, each sending the agent to a different “room” with deterministic high rewards of varying magnitudes.

This repository contains compact Gymnasium-style implementations of both environments for experimentation and reproducibility.

## RiverSwim Dynamics

In RiverSwim, the agent starts near the left end of a six-state chain (states 1 or 2, zero-indexed). At each step:

1. The agent picks an action — "swim left" (`0`) or "swim right" (`1`).
2. Swimming right succeeds with probability `p_right` (default `0.3`) and can be swept left with probability `p_left` (`0.1`), otherwise the agent stays put.
3. The agent receives a small reward of `5` when staying in the leftmost state after swimming left, and a large reward of `10 000` when remaining in the rightmost state after swimming right.

A key challenge is balancing the exploration of the risky high-reward state on the right versus the safer low-reward state on the left.

Below is a figure from the original paper showing the schematic of RiverSwim:

![RiverSwim Environment](docs/img/riverswim.png)

_(Image credit: Strehl et al., 2008)_

## SixArms Dynamics

SixArms consists of seven states. The agent begins in state `0` (the hub) and chooses one of six actions (arms):

- Each arm has a different success probability; on success, the agent moves to a dedicated room (states `1` through `6`).
- Inside each room, repeatedly selecting the matching action yields a deterministic high reward (from `50` up to `6000`), and the wrong action returns the agent to the hub without reward.
- The rarer the success probability, the larger the payoff, forcing the learner to balance exploration of low-probability, high-reward arms against faster but suboptimal options.

## Installation

You can install this environment using either **pip** or **uv** (a minimal package manager/distribution manager example in Python).
**Note**: If you're unfamiliar with `uv`, you can skip directly to the `pip` instructions.

### 1. Installing via `uv`

   ```bash
   uv add https://github.com/cruz-lucas/riverswim.git
   ```
   This should handle the necessary dependencies and set up the virtual environment if you have a `pyproject.toml` file, if not, see use `uv init`.

### 2. Installing via `pip`

   ```bash
   pip install git+https://github.com/cruz-lucas/riverswim.git
   ```
   This command installs the RiverSwim environment into your Python environment (**consider using a virtual environment**).

## Usage

Once installed, you can use the environment as follows:

```python
import gymnasium as gym
import riverswim  # Registers RiverSwim-v0 and SixArms-v0

env = gym.make("RiverSwim-v0")
obs, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

If you prefer, you can instantiate the classes directly (see `riverswim/river_swim_env.py` and `riverswim/six_arms_env.py` for the reference implementations).

## Citing

If you use this implementation for your research or reference, please cite the original paper:

```
@article{STREHL20081309,
title = {An analysis of model-based Interval Estimation for Markov Decision Processes},
journal = {Journal of Computer and System Sciences},
volume = {74},
number = {8},
pages = {1309-1331},
year = {2008},
note = {Learning Theory 2005},
issn = {0022-0000},
doi = {https://doi.org/10.1016/j.jcss.2007.08.009},
url = {https://www.sciencedirect.com/science/article/pii/S0022000008000767},
author = {Alexander L. Strehl and Michael L. Littman},
keywords = {Reinforcement learning, Learning theory, Markov Decision Processes},
abstract = {Several algorithms for learning near-optimal policies in Markov Decision Processes have been analyzed and proven efficient. Empirical results have suggested that Model-based Interval Estimation (MBIE) learns efficiently in practice, effectively balancing exploration and exploitation. This paper presents a theoretical analysis of MBIE and a new variation called MBIE-EB, proving their efficiency even under worst-case conditions. The paper also introduces a new performance metric, average loss, and relates it to its less “online” cousins from the literature.}
}
```

## Contributing

Contributions and suggestions to improve this implementation are always welcome. Feel free to open an issue or a pull request.

## License

This project is licensed under the [MIT License](LICENSE). Please see the [LICENSE](LICENSE) file for more information.
