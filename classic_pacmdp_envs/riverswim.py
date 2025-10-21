"""RiverSwim (Strehl & Littman, 2008) functional environment implemented with JAX."""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnasium import spaces
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv
from gymnasium.experimental.functional import FuncEnv
from gymnasium.utils import EzPickle
from gymnasium.wrappers import HumanRendering


class RenderStateType(NamedTuple):
    """Persistent render configuration for the RiverSwim environment."""

    background: np.ndarray
    cell_width: int
    height: int


class EnvState(NamedTuple):
    """Stateless RiverSwim environment state."""

    position: jax.Array


@struct.dataclass
class RiverSwimParams:
    """Default parameters for River Swim environment."""

    # Probabilities must add up to 1
    p_right: float = 0.3
    p_left: float = 0.1
    p_stay: float = 0.6

    hard_reward: int | float = 10_000
    easy_reward: int | float = 5
    common_reward: int | float = 0


PRNGKeyType: TypeAlias = jax.Array


class RiverSwimFunctional(
    FuncEnv[
        EnvState,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        RenderStateType,
        RiverSwimParams,
    ]
):
    """RiverSwim environment expressed through the functional Gymnasium API."""

    action_space = spaces.Discrete(2)
    observation_space = spaces.Discrete(6)

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
    }

    def initial(
        self, rng: PRNGKeyType, params: RiverSwimParams | None = None
    ) -> EnvState:
        """Sample an initial state."""
        start_position = jax.random.choice(rng, jnp.arange(1, 3), p=jnp.full((2,), 1 / 2))
        return EnvState(position=jnp.asarray(start_position, dtype=jnp.int32))

    def observation(
        self, state: EnvState, rng: PRNGKeyType, params: RiverSwimParams | None = None
    ) -> jax.Array:
        """Returns the current state index as the observation."""
        return jnp.asarray(state.position, dtype=jnp.int32)

    def transition(
        self,
        state: EnvState,
        action: jax.Array,
        rng: PRNGKeyType,
        params: RiverSwimParams | None = None,
    ) -> EnvState:
        """Sample the next state given the current state and action."""
        params = self.get_default_params()

        left_pos = jnp.maximum(state.position - 1, 0)
        right_pos = jnp.minimum(state.position + 1, 5)

        at_last_pos = jnp.equal(state.position, 5)
        candidates = jnp.stack([left_pos, state.position, right_pos])
        probabilities = jnp.where(
            at_last_pos,
            jnp.asarray(
                [1-params.p_right, 0.0, params.p_right], dtype=jnp.float32
            ),
            jnp.asarray(
                [params.p_left, params.p_stay, params.p_right], dtype=jnp.float32
            ),
        )     
        
        going_right_position = jax.random.choice(rng, candidates, p=probabilities)
        next_position = (1 - action) * left_pos + action * going_right_position

        return EnvState(position=jnp.asarray(next_position, dtype=jnp.int32))

    def reward(
        self,
        state: EnvState,
        action: jax.Array,
        next_state: EnvState,
        params: RiverSwimParams | None = None,
    ) -> jax.Array:
        """Compute the reward for a state transition."""
        params = self.get_default_params()

        same_state = jnp.equal(state.position, next_state.position)

        easy_reward = (1 - action) * jnp.equal(state.position, 0) * params.easy_reward
        hard_reward = (
            action * same_state * jnp.equal(state.position, 5) * params.hard_reward
        )

        reward = easy_reward + hard_reward
        return jnp.asarray(reward, dtype=jnp.float32)

    def terminal(
        self, state: EnvState, rng: PRNGKeyType, params: RiverSwimParams | None = None
    ) -> jax.Array:
        """Riverswim has no terminal states."""
        return jnp.asarray(False, dtype=jnp.bool_)

    def state_info(
        self, state: EnvState, params: RiverSwimParams | None = None
    ) -> dict[str, jax.Array]:
        """Return debugging info for the given state."""
        return {"position": jnp.asarray(state.position, dtype=jnp.int32)}

    def render_init(
        self,
        cell_width: int = 80,
        height: int = 120,
        background_color: tuple[int, int, int] = (240, 240, 240),
        start_color: tuple[int, int, int] = (250, 225, 225),
        goal_color: tuple[int, int, int] = (225, 250, 225),
        border_color: tuple[int, int, int] = (30, 30, 30),
    ) -> RenderStateType:
        """Initializes reusable rendering buffers."""
        width = cell_width * 6
        background = np.full((height, width, 3), background_color, dtype=np.uint8)

        background[:, :cell_width] = np.array(start_color, dtype=np.uint8)
        goal_slice = slice((6 - 1) * cell_width, 6 * cell_width)
        background[:, goal_slice] = np.array(goal_color, dtype=np.uint8)

        for idx in range(6 + 1):
            column = min(idx * cell_width, width - 1)
            background[:, column : column + 1] = np.array(border_color, dtype=np.uint8)

        return RenderStateType(
            background=background, cell_width=cell_width, height=height
        )

    def render_image(
        self,
        state: EnvState,
        render_state: RenderStateType,
        params: None = None,
    ) -> tuple[RenderStateType, np.ndarray]:
        """Render the current agent position as an RGB array."""
        frame = render_state.background.copy()
        agent_color = np.array((70, 130, 180), dtype=np.uint8)

        position = int(np.asarray(state.position))
        start_col = position * render_state.cell_width + 1
        end_col = max(start_col, (position + 1) * render_state.cell_width - 1)
        frame[:, start_col:end_col] = agent_color

        return render_state, frame

    def render_close(self, render_state: RenderStateType, params: None = None) -> None:
        """Nothing to clean up for the numpy-based renderer."""
        return None

    def get_default_params(self, **kwargs) -> RiverSwimParams:
        """Get the default params."""
        return RiverSwimParams(**kwargs)


class RiverSwimJaxEnv(FunctionalJaxEnv, EzPickle):
    """Gymnasium wrapper around the functional RiverSwim environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs):
        """Wraps functional environment."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)
        env = RiverSwimFunctional(**kwargs)
        env.transform(jax.jit)

        super().__init__(
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


if __name__ == "__main__":
    env = HumanRendering(RiverSwimJaxEnv(render_mode="rgb_array"))

    obs, info = env.reset()
    print(obs, info)

    terminal = False
    while not terminal:
        action = int(input("Please input an action\n"))
        obs, reward, terminal, truncated, info = env.step(action)
        print(obs, reward, terminal, truncated, info)

    exit()
