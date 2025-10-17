"""Pure JAX random policy rollout for the RiverSwim environment."""

from __future__ import annotations

import argparse
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv

from classic_pacmdp_envs import RiverSwimJaxEnv


class EpisodeStep(NamedTuple):
    """Structure to carry out the outcome of a step."""

    observation: jax.Array
    action: jax.Array
    next_observation: jax.Array
    reward: jax.Array
    terminal: jax.Array


class Episode(NamedTuple):
    """Structure to carry out the outcome of an episode."""

    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    rewards: jax.Array
    terminals: jax.Array
    total_return: jax.Array


@partial(jax.jit, static_argnames=("env", "num_steps"))
def rollout(env: FunctionalJaxEnv, rng: jax.Array, num_steps: int) -> Episode:
    """Run a single episode with random actions using only JAX primitives."""
    obs, _ = env.reset()

    def step_fn(carry, _):
        obs, rng = carry

        rng, action_key = jax.random.split(rng)

        action = jax.random.randint(
            action_key,
            shape=(),
            minval=0,
            maxval=env.action_space.n,
        )

        next_obs, reward, terminal, truncated, info = env.step(action)

        return (next_obs, rng), EpisodeStep(obs, action, next_obs, reward, terminal)

    (_, _), trajectory = jax.lax.scan(
        step_fn, (obs, rng), jnp.arange(num_steps, dtype=jnp.int32)
    )

    return Episode(
        observations=trajectory.observation,
        actions=trajectory.action,
        next_observations=trajectory.next_observation,
        rewards=trajectory.reward,
        terminals=trajectory.terminal,
        total_return=jnp.sum(trajectory.reward),
    )


@partial(jax.jit, static_argnames=("env", "num_steps"))
def batch_rollout(env: FunctionalJaxEnv, rngs: jax.Array, num_steps: int) -> Episode:
    """Vectorized rollout for multiple episodes."""

    def _run(key: jax.Array) -> Episode:
        return rollout(env, key, num_steps)

    return jax.vmap(_run)(rngs)


def run_random_policy(
    env: FunctionalJaxEnv,
    rng: jax.Array,
    num_episodes: int,
    episode_length: int,
) -> tuple[Episode, jax.Array, jax.Array]:
    """Execute random policy episodes and gather statistics."""
    rng, batch_key = jax.random.split(rng)
    rngs = jax.random.split(batch_key, num_episodes)

    episodes = batch_rollout(env, rngs, episode_length)
    returns = episodes.total_return
    mean_return = jnp.mean(returns)

    return episodes, returns, mean_return


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Pure JAX random policy example for RiverSwim."
    )
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--episode-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Main."""
    args = parse_args()
    env = RiverSwimJaxEnv()

    rng = jax.random.PRNGKey(args.seed)
    episodes, returns, mean_return = run_random_policy(
        env, rng, args.num_episodes, args.episode_length
    )

    print(
        f"Average return over {args.num_episodes} episodes: "
        f"{float(mean_return):.2f}"
    )
    print(f"Episode returns: {np.asarray(returns)}")

    sample = 0
    print(
        "Sampled observations:",
        np.asarray(episodes.observations[sample]),
    )
    print("Sampled actions:", np.asarray(episodes.actions[sample]))
    print("Sampled rewards:", np.asarray(episodes.rewards[sample]))


if __name__ == "__main__":
    main()
