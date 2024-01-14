import mss
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor
from .controller import Controller
from src.model.model import Model
from typing import List, Tuple, Optional
from PIL import Image


def _take_screenshot(monitor: int = 1, size: Tuple[int, int] = None) -> None:
    image = None
    with mss.mss() as sct:
        image = sct.grab(sct.monitors[monitor])
    
    image = Image.frombytes("RGB", image.size, image.bgra, "raw", "BGRX")
    if size is not None:
        image = image.resize(size)


class Bot:
    def __init__(self, keys: List[str], mouse_space: Tuple[int, int], visual_space: Tuple[int, int], window_space: Optional[Tuple[int, int]] = None, model: Model = None) -> None:
        self.controller = Controller(keys, mouse_space, window_space)
        self.model = model
        self.visual_space = visual_space
        
    def get_observation(self) -> Tensor:
        return pil_to_tensor(_take_screenshot(size=self.visual_space))
    
    def update(self, observation: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        if observation is None:
            observation = self.get_observation()
        
        action_tensors = self.model(observation)
        action_events = self.model.decode(action_tensors)
        
        self.controller.schedule_events(action_events)
        
        return action_tensors
    