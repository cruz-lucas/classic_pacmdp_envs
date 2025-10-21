"""SixArms (Strehl & Littman, 2008) functional environment implemented with JAX."""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv
from gymnasium.experimental.functional import FuncEnv
from gymnasium.utils import EzPickle
from gymnasium.wrappers import HumanRendering


class RenderStateType(NamedTuple):
    """Persistent render configuration for the SixArms environment."""

    background: np.ndarray
    cell_width: int
    height: int


class EnvState(NamedTuple):
    """Stateless SixArms environment state."""

    position: jax.Array


PRNGKeyType: TypeAlias = jax.Array


class SixArmsFunctional(
    FuncEnv[EnvState, jax.Array, jax.Array, jax.Array, jax.Array, RenderStateType, None]
):
    """SixArms environment expressed through the functional Gymnasium API."""

    action_space = spaces.Discrete(6)
    observation_space = spaces.Discrete(7)

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 1,
    }

    def initial(self, rng: PRNGKeyType, params=None):
        """Sample an initial state."""
        return EnvState(position=jnp.asarray(0, dtype=jnp.int32))

    def observation(self, state: EnvState, rng: PRNGKeyType, params=None) -> jax.Array:
        """Returns the current state index as the observation."""
        return jnp.asarray(state.position, dtype=jnp.int32)

    def transition(
        self,
        state: EnvState,
        action: jax.Array,
        rng: PRNGKeyType,
        params=None,
    ) -> EnvState:
        """Sample the next state given the current state and action."""
        key, _ = jax.random.split(rng, 2)

        position = jnp.asarray(state.position, dtype=jnp.int32)
        action = jnp.asarray(action, dtype=jnp.int32)

        p_all_arm = jnp.asarray([1.0, 0.15, 0.1, 0.05, 0.03, 0.01], dtype=jnp.float32)
        arm_success_prob = jnp.take(p_all_arm, action, mode="clip")

        candidates = jnp.stack([jnp.asarray(0, dtype=jnp.int32), action + 1])
        probabilities = jnp.stack([1.0 - arm_success_prob, arm_success_prob])
        next_from_center = jax.random.choice(key, candidates, p=probabilities)

        branch_transitions = jnp.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 1],
                [0, 2, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
                [0, 0, 0, 0, 5, 0],
                [0, 0, 0, 0, 0, 6],
            ],
            dtype=jnp.int32,
        )
        branch_row = jnp.take(branch_transitions, position, axis=0, mode="clip")
        next_from_branch = jnp.take(branch_row, action, axis=0, mode="clip")

        is_center = jnp.equal(position, 0)
        next_position = jnp.where(is_center, next_from_center, next_from_branch)

        return EnvState(position=jnp.asarray(next_position, dtype=jnp.int32))

    def reward(
        self,
        state: EnvState,
        action: jax.Array,
        next_state: EnvState,
        params=None,
    ) -> jax.Array:
        """Compute the reward for a state transition."""
        same_state = jnp.equal(state.position, next_state.position)
        rewards = jnp.asarray([0, 50, 133, 300, 800, 1660, 6000], dtype=jnp.float32)
        reward_values = jnp.take(rewards, next_state.position, mode="clip")
        reward = reward_values * same_state.astype(jnp.float32)
        return reward

    def terminal(self, state: EnvState, rng: PRNGKeyType, params=None) -> jax.Array:
        """Sixarms has no terminal states."""
        return jnp.asarray(False, dtype=jnp.bool_)

    def state_info(self, state: EnvState, params=None) -> dict[str, jax.Array]:
        """Return debugging info for the given state."""
        return {"position": jnp.asarray(state.position, dtype=jnp.int32)}

    def render_init(
        self,
        cell_width: int = 90,
        height: int = 140,
        background_color: tuple[int, int, int] = (240, 240, 240),
        center_color: tuple[int, int, int] = (220, 240, 250),
        arm_color: tuple[int, int, int] = (245, 230, 220),
        border_color: tuple[int, int, int] = (30, 30, 30),
    ) -> RenderStateType:
        """Initializes reusable rendering buffers."""
        width = cell_width * 7
        background = np.full((height, width, 3), background_color, dtype=np.uint8)

        # Highlight the central hub (state 0).
        background[:, :cell_width] = np.array(center_color, dtype=np.uint8)

        # Highlight the six arms.
        arm_slice_color = np.array(arm_color, dtype=np.uint8)
        for idx in range(1, 7):
            cell_slice = slice(idx * cell_width, (idx + 1) * cell_width)
            background[:, cell_slice] = arm_slice_color

        # Draw vertical borders between cells.
        border = np.array(border_color, dtype=np.uint8)
        for idx in range(7 + 1):
            column = min(idx * cell_width, width - 1)
            background[:, column : column + 1] = border

        return RenderStateType(
            background=background, cell_width=cell_width, height=height
        )

    def render_image(
        self,
        state: EnvState,
        render_state: RenderStateType,
        params=None,
    ) -> tuple[RenderStateType, np.ndarray]:
        """Render the current agent position as an RGB array."""
        frame = render_state.background.copy()
        agent_color = np.array((70, 130, 180), dtype=np.uint8)

        position = int(np.asarray(state.position))
        position = max(0, min(6, position))

        start_col = position * render_state.cell_width + 1
        end_col = max(start_col, (position + 1) * render_state.cell_width - 1)
        frame[:, start_col:end_col] = agent_color

        return render_state, frame

    def render_close(self, render_state, params=None) -> None:
        """Nothing to clean up for the numpy-based renderer."""
        return None


class SixArms(FunctionalJaxEnv, EzPickle):
    """Gymnasium wrapper around the functional SixArms environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs):
        """Wraps functional environment."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)
        env = SixArmsFunctional(**kwargs)
        env.transform(jax.jit)

        super().__init__(
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


if __name__ == "__main__":
    env = HumanRendering(SixArms(render_mode="rgb_array"))

    obs, info = env.reset()
    print(obs, info)

    terminal = False
    while not terminal:
        action = int(input("Please input an action\n"))
        obs, reward, terminal, truncated, info = env.step(action)
        print(obs, reward, terminal, truncated, info)

    exit()
