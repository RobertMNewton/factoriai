import sys
import os

# Get the absolute path of the 'factorai' directory
factorai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the 'factorai' directory to the Python path
sys.path.insert(0, factorai_path)

import torch

from src.rl.train_rl import rl_train_loop
from src.rl.reward import HumanFeedbackReward
from src.model.model import Default
from src.config import default_config

run_name = "test_rl"

if __name__ == "__main__":
    model = Default(
        default_config.keys,
        default_config.delays,
        default_config.mouse_space,
        default_config.window_space,
    )
    
    model.load_state_dict(torch.load(f"runs/{run_name}/ckpt.pt"))
    
    reward_fn = HumanFeedbackReward(0.1, "p")
    
    rl_train_loop(model, reward_fn)
    
    if not os.path.exists(f"runs/{run_name}/rl/ckpt.pt"):
        os.mkdir(f"runs/{run_name}/rl/ckpt.pt")
    
    model.save(run_name, dir="runs")
    