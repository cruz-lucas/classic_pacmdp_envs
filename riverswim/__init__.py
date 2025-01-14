"""Registers the internal gym envs then loads the env plugins for module using the entry point."""

from gymnasium.envs.registration import register


register(
    id="RiverSwim-v0",
    entry_point="riverswim.riverswim:RiverSwimEnv",
)
