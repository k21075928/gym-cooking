from gym.envs.registration import register

register(
        id="overcookedEnv-v0",
        entry_point="gym_cooking.envs:OvercookedEnvironment",
        )
register(
        id="DQLovercookedEnv-v0",
        entry_point="gym_cooking.DQL:DQLOvercookedEnvironment",
        )