import os
import torch
from torch import nn, Tensor
from torch.nn import Module
from typing import Tuple, List, Optional, Dict, Any, Callable

from .modules import VisionModule, MemoryModule, ActionModule, \
    DelayModule, KeystrokeModule, MouseModule
    
from .vision_models import vgg
from .memory_models import transformer_memory
from .action_models import action_transformer
from .mouse_models import deconv_vgg
from .mlp_models import mlp


class Model(Module):
    """
    Model ties together all sub-models in the architecture. Each sub-model needs to implement
    all of its functionality so that its interface can be reduced to an input-output paradigm.
    """
    def __init__(
            self,
            vision_encoder:     VisionModule,
            memory_network:     MemoryModule,
            action_network:     ActionModule,
            delay_network:      DelayModule,
            keystroke_network:  KeystrokeModule,
            mouse_networks:     MouseModule,
    ):
        super(Model, self).__init__()

        self.vision_encoder = vision_encoder
        self.memory_network = memory_network
        self.action_network = action_network
        self.delay_network = delay_network
        self.keystroke_network = keystroke_network
        self.mouse_network = mouse_networks
    
    def forward(self, image: Tensor, train: Optional[int] = None, max_actions: int = 50) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns keystrokes, delays and mouse_positions encoded probabilistically as action tensors that can be decoded
        for scheduling bot actions.
        
        For un-batched inputs, 1st dim should be each action. For batched inputs, 2nd dim should be the actions
        """
        image_enc: Tensor = self.vision_encoder(image)
        memory_enc: Tensor = self.memory_network(image_enc)
        actions, end_token_prob = self.action_network(memory_enc, train=train, max_tokens=max_actions)
        
        keystrokes, keystroke_encs = self.keystroke_network(actions)
        delays, delay_encs = self.delay_network(keystroke_encs)
        
        augmented_image_enc = torch.cat((delay_encs, image_enc.expand(delay_encs.shape[0], -1)), dim=-1)
        mouse_positions: Tensor = self.mouse_network(augmented_image_enc)

        if train is not None:
            return keystrokes, delays, mouse_positions, end_token_prob
        else:
            return keystrokes, delays, mouse_positions
    
    
    def decode(self, actions: List[Tuple[Tensor, Tensor, Tensor]]) -> List[Tuple[str, int, Tuple[int, int]]]:
        res = []
        for action in actions:
            keystrokes, delays, mouse_positions = action
            res.append(
                (
                    self.keystroke_network.decode(keystrokes),
                    self.delay_network.decode(delays),
                    self.mouse_network.decode(mouse_positions),
                )
            )
            
    def reset_memory(self) -> None:
        self.memory_network.reset_memory()
        
    def get_size(self) -> int:
        return sum(
            [
                self.vision_encoder.get_size(),
                self.memory_network.get_size(),
                self.action_network.get_size(),
                self.delay_network.get_size(),
                self.keystroke_network.get_size(),
                self.mouse_network.get_size(),
            ]
        )
        
    def save(self, session: str, name: str = "ckpt", dir: str = "runs") -> None:
        path = f"{dir}/{session}/{name}.pt"
        
        tmp = ""
        for dir in path.split("/")[:-1]:
            tmp += "/" + dir
            if not os.path.exists(tmp[1:]):
                os.mkdir(tmp[:1])
                
        torch.save(self.state_dict(), path)


def _default_networks_from(
    action_space: List[str],
    delay_space: List[int],
    mouse_space: Tuple[int, int], 
    visual_space: Tuple[int, int],
    visual_encoding_size: int,
    memory_encoding_size: int,
    action_encoding_size: int,
    peripheral_encoding_size: int,
    memory_size: int = 256,
    visual_network: Callable = vgg.Mini,
    memory_network: Callable = transformer_memory.Mini,
    action_network: Callable = action_transformer.Mini,
    delay_network: Callable = mlp.BaseDelayClassifier,
    keystroke_network: Callable = mlp.BaseKeystrokeClassifier,
    mouse_network: Callable = deconv_vgg.Mini,
    ) -> Tuple[VisionModule, MemoryModule, ActionModule, DelayModule, KeystrokeModule, MouseModule]:
    """
    Returns VisionModule, MemoryModule, ActionModule, DelayModule, KeystrokeModule, MouseModule
    """
    return (
        VisionModule(
            [(3, *visual_space)],
            [(1, visual_encoding_size)],
            network=visual_network(
                input_dims=visual_space,
                mlp_output_dims=visual_encoding_size,
        )),
        MemoryModule(
            [(1, visual_encoding_size,)],
            [(1, memory_encoding_size)], 
            network=memory_network(
                input_feature_dims=visual_encoding_size,
                output_feature_dims=memory_encoding_size,
                memory_size=memory_size,
        )),
        ActionModule(
            [(1, memory_encoding_size)],
            [(-1, action_encoding_size), (-1, 2)],            
            network=action_network(
                input_feature_dims=memory_encoding_size,
                output_feature_dims=action_encoding_size,
            )),
        DelayModule(
            [(-1, peripheral_encoding_size)],
            delays=delay_space,
            encoding_size=peripheral_encoding_size,
            network=delay_network(
                peripheral_encoding_size,
                delay_space,
                peripheral_encoding_size,
        )),
        KeystrokeModule(
            [(-1, action_encoding_size)],
            keys=action_space,
            encoding_size=peripheral_encoding_size,
            network=keystroke_network(
                action_encoding_size,
                action_space,
                peripheral_encoding_size,
        )),
        MouseModule(
            [(-1, visual_encoding_size+peripheral_encoding_size)],
            [(-1, 1, *mouse_space)],
            network=mouse_network(
                input_dims=peripheral_encoding_size + visual_encoding_size,
                output_dims=mouse_space,
        )),
    )


class Default(Model):
    def __init__(
        self,
        action_space: List[str],
        delay_space: List[int],
        mouse_space: Tuple[int, int], 
        visual_space: Tuple[int, int]
        ):
        super().__init__(*_default_networks_from(
            action_space=action_space,
            delay_space=delay_space,
            mouse_space=mouse_space,
            visual_space=visual_space,
            visual_encoding_size=256,
            memory_encoding_size=256,
            action_encoding_size=256,
            peripheral_encoding_size=256,
        ))