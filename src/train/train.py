from .data_loader import load_data, get_sessions, get_n_steps
from ..model.model import Model, Default
from .logger import Log, new_entry, Config, default_config

from .. import utils

import torch
from torch import optim, nn
from torch.optim import Optimizer

from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel
from tqdm import tqdm
from functools import partial


def train_loop(model: Model, epochs: int, lr: float, optimiser: Optimizer = optim.Adam, config: Config = default_config, dir="data", device = None, criterion: nn.Module = nn.MSELoss(), verbose: bool = True, log: Optional[Log] = None, chkpt_steps: int = 10, reset_steps: Optional[int] = None) -> None:
    """
    Trains model in place
    """
    optimiser = optimiser(model.parameters(), lr=lr)
    
    sessions = get_sessions(dir=dir)
    n_steps = get_n_steps(sessions, dir=dir)
    
    if device is None:
        device = utils.get_device()
    else:
        utils.set_device(device)
            
    model = model.to(config.get_dtype())
    model = model.to(device)
    
    get_data = partial(
        load_data,
        keys=config.keys,
        delays=config.delays,
        scrolls=config.scrolls,
        mouse_space=config.mouse_space,
        dtype=config.get_dtype(),
        device=device,
        dir=dir,
    )
    
    start_epoch, start_step, started = 0, 0, True
    if len(log.entries) > 0:
        start_epoch = log.entries[-1].epoch
        start_step = log.entries[-1].step
        started = False
    
    for epoch in range(start_epoch, epochs):
        last_labels = None
        step, running_loss = 0, 0
        for si, session in enumerate(sessions):
            if reset_steps is None:
                model.reset_memory()
            # pbar = tqdm(get_data(session=session), desc="step", total=n_steps)
            # for features, labels in pbar:
            for features, labels in get_data(session=session):
                if started:
                    if last_labels is not None:    
                        num_tokens = last_labels[0].shape[0] if len(last_labels) > 0 else 1
                        preds = model(features, train=num_tokens)
                        
                        loss = None
                        if last_labels == []:
                            loss = criterion(preds[-1], torch.Tensor([[0, 1]]))
                        else: 
                            end_token_labels = torch.zeros((last_labels[0].shape[0], 2), device=utils.get_device())
                            for i in range(num_tokens):
                                end_token_labels[i, 0] = 1.0
                            end_token_labels[-1, 1] = 1.0
                            
                            loss = sum(criterion(pred, label) for pred, label in zip(preds, last_labels + [end_token_labels]))
                        
                        loss.backward()
                        running_loss += loss
                        
                        optimiser.step()
                        optimiser.zero_grad()
                        
                        torch.cuda.empty_cache()
                        
                        if reset_steps is not None and step % reset_steps == 0:
                            model.reset_memory()
                        
                        # bar.set_description(f"loss: {loss / step}")
                        print(f"epoch: {epoch}, step: {step}, loss: {loss:.5f}")
                        
                        if log is not None:
                            log.add_entry(
                                new_entry(
                                    step,
                                    epoch,
                                    loss=running_loss/step
                                )
                            )
                    
                    step  += 1
                    last_labels = labels 
                    
                    if step % chkpt_steps == 0 and log is not None:
                        log.save()
                        model.save(log.session)
                else:
                    step  += 1
                    
                    if step == start_step:
                        started = True
                 
            
            