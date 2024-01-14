import torch
import os
import json

from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import Optional, List, Tuple
from .data_loader import RMB, LMB, SCROLL

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

class LogEntry(BaseModel):
    step: int
    epoch: int
    
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    
    timestamp: Optional[int] = None # this should be in milliseconds
    
class Log(BaseModel):
    entries: List[LogEntry]
    
    session: str
    date: str
    config: Config
    
    def add_entry(self, entry: LogEntry) -> None:
        self.entries.append(entry)
        
    def save(self, dir: str = "runs") -> None:
        log, path = self.model_dump(), f"{dir}/{self.session}/log.json"
        
        temp = ""
        for folder in path.split("/")[:-1]:
            temp += "/" + folder
            if not os.path.exists(temp[1:]):
                os.mkdir(temp[1:])
        
        with open(path, "w") as f:
            json.dump(log, f)
            
            
def new_log(session: str, config: Config) -> Log:
    return Log(
        entries=[],
        session=session,
        date=str(datetime.now()),
        config=config
    )
    
def new_entry(step: int, epoch: int, loss: Optional[float] = None, accuracy: Optional[float] = None):
    return LogEntry(
        step=step,
        epoch=epoch,
        loss=loss,
        accuracy=accuracy,
        timestamp=int(datetime.now().timestamp() * 1000),
    )
            