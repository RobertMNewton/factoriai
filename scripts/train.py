import sys
import os

# Get the absolute path of the 'factorai' directory
factorai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the 'factorai' directory to the Python path
sys.path.insert(0, factorai_path)


import torch
from src.train.train import train_loop
from src.config import default_config
from src.train.logger import new_log, Log, from_file
from src.model.model import Default
from src import utils


run_name = "test"

if __name__ == "__main__":
    utils.set_device(torch.device("cpu"))
    
    model = Default(
        default_config.keys,
        default_config.delays,
        default_config.mouse_space,
        default_config.window_space,
    )
    
    log = None
    if os.path.exists(f"runs/{run_name}"):
        model.load_state_dict(torch.load(f"runs/{run_name}/ckpt.pt"))
        log = from_file(f"runs/{run_name}/log.json")
    else:
        log = new_log("test", default_config)
    
    print(f"Model Size: {model.get_size()}")
    
    train_loop(model, 10, 3E-5, device=torch.device("cpu"), log=log, reset_steps=160, skip_still_frames=True)
