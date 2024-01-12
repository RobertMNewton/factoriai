from torch import nn, Tensor
from torch.nn import Module
from typing import Tuple, List, Optional

from model.modules import VisionModule, MemoryModule, ActionModule, \
    DelayModule, KeystrokeModule, MouseModule


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
    
    def forward(self, image: Tensor, train: Optional[int] = None, max_actions: int = 20) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns keystrokes, delays and mouse_positions encoded probabilistically as action tensors that can be decoded
        for scheduling bot actions.
        
        For un-batched inputs, 1st dim should be each action. For batched inputs, 2nd dim should be the actions
        """
        image_enc: Tensor = self.vision_encoder(image)
        memory_enc: Tensor = self.memory_network(image_enc)
        actions, end_token_prob: Tuple[Tensor, Tensor] = self.action_network(memory_enc, train=train, max_actions=max_actions)
        
        keystrokes, keystroke_encs: Tuple[Tensor, Tensor] = self.keystroke_network(actions)
        delays, delay_encs: Tuple[Tensor, Tensor] = self.delay_network(actions, keystroke_encs)
        mouse_positions: Tensor = self.mouse_network(actions, delay_encs, image_enc)

        if train is not None:
            return keystrokes, delays, mouse_positions, end_token_prob
        else:
            return keystrokes, delays, mouse_positions
    
    
    def decode(self, actions: List[Tuple[Tensor, Tensor, Tensor]]) -> List[Tuple[List[str], int, Tuple[int, int]]]:
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
        
