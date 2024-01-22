import torch
from pydantic import BaseModel
from typing import List, Optional, Tuple
from src.train.data_loader import RMB, LMB, SCROLL

class Config(BaseModel):
    keys: List[Optional[str]]
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
    keys=["a", "w", "s", "d", "e", "c", "z", "shift", RMB, LMB, None],
    delays=list(range(10, 260, 10)),
    scrolls=list(range(-5, 6)),
    mouse_space=(384, 384),
    window_space=(400, 400),
    dtype="float32"
)
