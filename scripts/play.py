import sys
import os

# Get the absolute path of the 'factorai' directory
factorai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the 'factorai' directory to the Python path
sys.path.insert(0, factorai_path)

import torch
import time

from src.config import default_config
from src import utils
from src.model.model import Default
from src.bot.bot import Bot


if __name__ == "__main__":
    utils.set_device(torch.device("cpu"))
    
    model = Default(
        default_config.keys,
        default_config.delays,
        default_config.mouse_space,
        default_config.window_space,
    )
    model.load_state_dict(torch.load("runs/test/ckpt.pt"))
    model = model.to(utils.get_device())
    
    bot = Bot(
        model,
        default_config.keys,
        default_config.mouse_space,
        default_config.window_space,
    )
    
    start = time.time()
    while time.time() < start + 4*60*10:
        bot.step()
        time.sleep(0.25)
    