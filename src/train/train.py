from data_loader import load_data, get_sessions
from data_loader import RMB, LMB, SCROLL
from model.model import Model, Default

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
    dtype: torch.dtype = torch.float32
    
    def __init__(self, **data) -> None:
        super(Config, self).__init__(**data)
        
        for scroll in self.scrolls:
            self.keys.append(f"{SCROLL}{scroll}")
    

default_config = Config(
    keys=["a", "w", "s", "d", "e", "c", "z", "shift", RMB, LMB],
    delays=list(range(25, 275, 25)),
    scrolls=list(range(-5, 6)),
    mouse_space=(400, 400),
    window_space=(400, 400),
)

def train(model: Model, epochs: int, lr: float, optimiser: Optimizer = optim.Adam, config: Config = default_config, dir="data", device = None, loss: nn.Module = nn.MSELoss) -> None:
    """
    Trains model in place
    """
    optimiser = optimiser(model.parameters(), lr=lr)
    sessions = get_sessions(dir=dir)
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
    model.to(config.dtype)
    model.to(device)
    
    load_data = partial(
        load_data,
        keys=config.keys,
        delays=config.delays,
        scrolls=config.scrolls,
        mouse_space=config.mouse_space,
        dtype=torch.dtype,
        device=device,
        dir=dir,
    )
    
    for epoch in range(epochs):
        for si, session in enumerate(sessions):
            for features, labels in tqdm(load_data(session=session)):
                pred = model(features)
                
                loss = loss(pred, labels)
                loss.backward()
                
                optimiser.step()

# mini test run
model = Default(
    default_config.keys,
    default_config.delays,
    default_config.mouse_space,
    default_config.window_space,
)

train(model, 5, 3E-6)      
            
            