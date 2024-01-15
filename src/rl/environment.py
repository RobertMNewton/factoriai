import time
from torch import Tensor
from gym import Env, spaces
from .reward import AbstractReward
from src.bot.bot import Bot
from typing import List, Tuple, Any

def _wait_until(ts: float) -> int:
    """
    waits until given timestamp has passed. Returns current posix timestamp at function return
    """
    while time.time() < ts:
        time.sleep(0.025)
    return time.time()


class FactorioEnv(Env):
    def __init__(self, bot: Bot, reward_fn: AbstractReward, keys: List[str], delays: List[int], mouse_space: Tuple[int, int], episode_time: int = 60*5) -> None:
        super().__init__()
        
        self.bot = bot
        self.reward_fn = AbstractReward
        
        self.last_ts = None
        
        self.action_space = spaces.Sequence(
            spaces.Tuple((
                spaces.Box(low=0.0, high=1.0, shape=(1, len(keys))),
                spaces.Box(low=0.0, high=1.0, shape=(1, len(delays))),
                spaces.Box(low=0.0, high=1.0, shape=(1, *mouse_space)),
            ))
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=bot.visual_space)
        
        self.episode_duration = episode_time
        self.episode_end = time.time() + self.episode_duration
      
    def reset(self) -> Tuple[Tensor, dict]:
        self.last_ts, self.episode_end = time.time(),  time.time() + self.episode_duration
        return self.bot.get_observation(), {}
    
    def step(self, action: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[Tensor, float, bool, bool, dict]:
        self.bot.step_from(action)   
        
        self.last_ts = _wait_until(self.last_ts+0.25)
        return self.bot.get_observation(), self.reward_fn.get_reward(), self.episode_end <= time.time(), False, {}
      
    
      
      
      