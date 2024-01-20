import sys
import os

# Get the absolute path of the 'factorai' directory
factorai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the 'factorai' directory to the Python path
sys.path.insert(0, factorai_path)

# Now you can import your 'src' module
from src.train.train import train_loop, default_config

# Rest of your code here


import torch
from src.train.train import train_loop, default_config
from src.train.logger import new_log, Log
from src.model.model import Default
from src import utils


if __name__ == "__main__":
    utils.set_device(torch.device("cpu"))
    
    model = Default(
        default_config.keys,
        default_config.delays,
        default_config.mouse_space,
        default_config.window_space,
    )
    
    log = new_log("test", default_config)
    
    print(f"Model Size: {model.get_size()}")
    
    train_loop(model, 5, 6e-5, device=torch.device("cpu"), log=log)
