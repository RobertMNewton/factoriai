import torch
import os
import json

from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import Optional, List, Tuple

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
    
def from_file(path: str) -> Log:
    obj = json.load(open(path, "r"))
    return Log(**obj)
        
def new_entry(step: int, epoch: int, loss: Optional[float] = None, accuracy: Optional[float] = None):
    return LogEntry(
        step=step,
        epoch=epoch,
        loss=loss,
        accuracy=accuracy,
        timestamp=int(datetime.now().timestamp() * 1000),
    )

            