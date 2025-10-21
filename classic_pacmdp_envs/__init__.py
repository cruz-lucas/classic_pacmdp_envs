"""Registers the Strehl & Littman (2008) benchmark environments."""

from gymnasium.envs.registration import register

from classic_pacmdp_envs.riverswim import RiverSwim, RiverSwimJaxEnv
from classic_pacmdp_envs.sixarms import SixArms


register(
    id="RiverSwim-v1",
    entry_point="classic_pacmdp_envs.riverswim:RiverSwim",
)
register(
    id="RiverSwimJax-v1",
    entry_point="classic_pacmdp_envs.riverswim:RiverSwimJaxEnv",
)

register(
    id="SixArms-v1",
    entry_point="classic_pacmdp_envs.sixarms:SixArms",
)

__all__ = ["RiverSwim", "RiverSwimJaxEnv", "SixArms"]
