import torch


device = None
def get_device() -> torch.device:
    global device
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return device

def set_device(d: torch.device) -> None:
    global device
    device = d
    
    