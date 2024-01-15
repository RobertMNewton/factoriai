import os
from torch import Tensor
from abc import ABC, abstractmethod
from pynput.keyboard import Listener, Key
from typing import Optional, Callable


class AbstractReward(ABC):
    @abstractmethod
    def get_reward(self, *args, **kwargs) -> float:
        raise NotImplementedError()
    

class HumanFeedbackReward(AbstractReward):
    """
    Gets reward from human feedback
    """
    def __init__(self, reward: float, key: str | Key, start: bool = True) -> None:
        super().__init__()
        
        self.reward = reward
        self.cumulative_reward = 0
        self.key = key
        
        self.listener = Listener(self.add_reward)
        
        if start:
            self.start()
        
    def add_reward(self, key_pressed: Key) -> None:
        if key_pressed == self.key: 
            self.cumulative_reward += self.reward
        
    def start(self) -> None:
        self.listener.start()
    
    def stop(self) -> None:
        self.listener.stop()
        
    def get_reward(self, *args, **kwargs) -> float:
        reward = self.cumulative_reward
        self.cumulative_reward = 0
        
        return reward
    

class FactorioFileReward(AbstractReward):
    """
    Gives reward based on reward inside of Factorio mod written file (only way out of the Factorio sandbox)
    """
    def __init__(self, path: str) -> None:
        super().__init__()
        
        self.path = path
    
    def get_reward(self, *args, **kwargs) -> float:
        reward = 0
        
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                reward = float(f.read())
        return reward
    