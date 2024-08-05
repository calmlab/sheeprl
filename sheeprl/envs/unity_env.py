import os

from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper


class UnityWrapper(UnityToGymWrapper):
    def __init__(self, file_name, env_num_id) -> None:
        print('initailizing!')
        os.environ['DISPLAY'] = ':1'
        print('env_num_id:', env_num_id)
        print('base_port:',5005+env_num_id)
        print('file name:',file_name)
        env = UnityEnvironment(file_name, worker_id=env_num_id)
        super().__init__(env)

    # render_mode는 무조건 rgb이지만,
    @property
    def render_mode(self):
        return 'rgb_array'

    def _convert_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {"rgb": obs}


    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        if obs.dtype == np.float32:
            obs = (obs * 255).astype(np.uint8)
        return obs, reward, done, False, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs = super().reset()
        if obs.dtype == np.float32:
            obs = (obs * 255).astype(np.uint8)
        return obs, {}

