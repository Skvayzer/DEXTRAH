"""Gym registration for G1 ADEPT/SAPG with SimToolReal objects."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .g1_revo2_adept_env_cfg import G1Revo2AdeptEnvCfg
from .g1_revo2_direct_env import G1Revo2DirectEnvCfg
from .g1_revo2_bps_env import G1Revo2BpsEnvCfg
from .g1_revo2_touch_env import G1Revo2TouchEnvCfg


_PACKAGE_DIR = Path(__file__).resolve().parent

gym.register(
    id="G1-Revo2-SimToolReal-Repose-BPS128-Touch",
    entry_point="dextrah_lab.tasks.g1_revo2_adept.g1_revo2_touch_env:G1Revo2TouchEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1Revo2TouchEnvCfg,
        "rl_games_cfg_entry_point": "dextrah_lab.tasks.g1_revo2_adept.agents.touch_sapg:config",
    },
)

gym.register(
    id="G1-Revo2-SimToolReal-Repose-BPS128",
    entry_point="dextrah_lab.tasks.g1_revo2_adept.g1_revo2_bps_env:G1Revo2BpsEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1Revo2BpsEnvCfg,
        "rl_games_cfg_entry_point": str(_PACKAGE_DIR / "agents" / "g1_revo2_bps128_sapg.yaml"),
    },
)

gym.register(
    id="Adept-G1-Revo2-SimToolReal-Repose",
    entry_point=(
        "dextrah_lab.tasks.g1_revo2_adept.g1_revo2_adept_env:"
        "G1Revo2AdeptEnv"
    ),
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1Revo2AdeptEnvCfg,
        "rl_games_cfg_entry_point": str(
            _PACKAGE_DIR / "agents" / "g1_revo2_adept_sapg.yaml"
        ),
    },
)

gym.register(
    id="G1-Revo2-SimToolReal-Repose-Direct",
    entry_point="dextrah_lab.tasks.g1_revo2_adept.g1_revo2_direct_env:G1Revo2DirectEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1Revo2DirectEnvCfg,
        "rl_games_cfg_entry_point": str(_PACKAGE_DIR / "agents" / "g1_revo2_direct_sapg.yaml"),
    },
)
