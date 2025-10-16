"""Registers the Strehl & Littman (2008) benchmark environments."""

from gymnasium.envs.registration import register

from classic_pacmdp_envs.riverswim import RiverSwimJaxEnv
from classic_pacmdp_envs.sixarms import SixArmsJaxEnv


register(
    id="RiverSwim-v1",
    entry_point="classic_pacmdp_envs.riverswim:RiverSwimJaxEnv",
)

register(
    id="SixArms-v1",
    entry_point="classic_pacmdp_envs.sixarms:SixArmsJaxEnv",
)

__all__ = ["RiverSwimJaxEnv", "SixArmsJaxEnv"]
