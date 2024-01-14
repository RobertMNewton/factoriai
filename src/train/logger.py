import os
import json

from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import Optional, List

from .train import Config

class LogEntry(BaseModel):
    step: int
    epoch: int
    
    Loss: Optional[float]
    Accuracy: Optional[float]
    
    timestamp: Optional[int]  # this should be ms
    
class Log(BaseModel):
    entries: List[LogEntry]
    
    session: str
    date: datetime
    config: Config
    
    def add_entry(self, entry: LogEntry) -> None:
        self.entries.append(entry)
        
    def save(self, dir: str = "runs") -> None:
        log, path = self.model_dump(), f"{dir}/{self.session}/log.json"
        
        temp = ""
        for folder in path.split("/")[:-1]:
            temp += "/" + folder
            if not os.path.exists(temp):
                os.mkdir(temp)
        
        with open(path, "w") as f:
            json.dump(log, f)
            
            
def new_log(session: str, config: Config) -> Log:
    return Log(
        entries=[],
        session=session,
        date=datetime.now(),
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
            