import gymnasium as gym

from . import agents

##
# Register Gym environments/
##

gym.register(
    id="Inspire-Manip-v0",
    entry_point=f"{__name__}.inspire_manip_env:InspireManipEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inspire_manip_env_cfg:InspireManipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InspireManipPPORunnerCfg",
        # Optional: Add other RL libraries if needed
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)