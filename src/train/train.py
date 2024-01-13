from .data_loader import load_data, get_sessions, get_n_steps
from .data_loader import RMB, LMB, SCROLL
from ..model.model import Model, Default

from .. import utils

import torch
from torch import optim, nn
from torch.optim import Optimizer

from typing import Dict, Any, List, Tuple
from pydantic import BaseModel
from tqdm import tqdm
from functools import partial


class Config(BaseModel):
    keys: List[str]
    delays: List[int]
    scrolls: List[int]
    mouse_space: Tuple[int, int]
    window_space: Tuple[int, int]
    dtype: str = "float64"
    
    def __init__(self, **data) -> None:
        super(Config, self).__init__(**data)
        
        for scroll in self.scrolls:
            self.keys.append(f"{SCROLL}{scroll}")
            
    def get_dtype(self) -> torch.dtype:
        match self.dtype:
            case "float64":
                return torch.float64
            case "float32":
                return torch.float32
    

default_config = Config(
    keys=["a", "w", "s", "d", "e", "c", "z", "shift", RMB, LMB],
    delays=list(range(25, 275, 25)),
    scrolls=list(range(-5, 6)),
    mouse_space=(384, 384),
    window_space=(400, 400),
    dtype="float32"
)

def train_loop(model: Model, epochs: int, lr: float, optimiser: Optimizer = optim.Adam, config: Config = default_config, dir="data", device = None, criterion: nn.Module = nn.MSELoss(), verbose: bool = True) -> None:
    """
    Trains model in place
    """
    optimiser = optimiser(model.parameters(), lr=lr)
    
    sessions = get_sessions(dir=dir)
    n_steps = get_n_steps(sessions, dir=dir)
    
    if device is None:
        device = utils.get_device()
    else:
        utils.set_device(device)
            
    model = model.to(config.get_dtype())
    model = model.to(device)
    
    get_data = partial(
        load_data,
        keys=config.keys,
        delays=config.delays,
        scrolls=config.scrolls,
        mouse_space=config.mouse_space,
        dtype=config.get_dtype(),
        device=device,
        dir=dir,
    )
    
    
    for epoch in range(epochs):
        last_labels = None
        for si, session in enumerate(sessions):
            pbar = tqdm(get_data(session=session), desc="step", total=n_steps)
            for features, labels in pbar:
                if last_labels is not None:    
                    preds = model(features, train=len(last_labels))
                    
                    print(last_labels)
                    print(pred)
                    
                    loss = sum(criterion(pred, label) for pred, label in zip(preds, last_labels))
                    loss.backward()
                    
                    optimiser.step()
                    
                    pbar.set_description(f"loss: {loss.value}")
                
                last_labels = labels  
            
            