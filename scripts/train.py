import torch
from src.train.train import train_loop, default_config
from src.utils import new_log, Log
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
    
    train_loop(model, 5, 3e-3, device=torch.device("cpu"), log=log)
