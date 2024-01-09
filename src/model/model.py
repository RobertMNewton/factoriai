from torch import nn, Tensor
from torch.nn import Module
from typing import Tuple


class Model(Module):
    """
    Model ties together all sub-models in the architecture. Each sub-model needs to implement
    all of its functionality so that its interface can be reduced to an input-output paradigm.
    """
    def __init__(
            self,
            vision_encoder:     Module,
            memory_network:      Module,
            action_network:      Module,
            delay_network:       Module,
            keystroke_network:   Module,
            mouse_networks:      Module,
    ):
        super(Model, self).__init__()

        self.vision_encoder = vision_encoder
        self.memory_network = memory_network
        self.action_network = action_network
        self.delay_network = delay_network
        self.keystroke_network = keystroke_network
        self.mouse_network = mouse_networks
    
    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns keystrokes, delays and mouse_positions encoded probabilistically as action tensors that can be decoded
        for scheduling bot actions.
        
        For un-batched inputs, 1st dim should be each action. For batched inputs, 2nd dim should be the actions
        """
        image_enc: Tensor = self.vision_encoder(image)
        memory_enc: Tensor = self.memory_network(image_enc)
        actions: Tensor = self.action_network(memory_enc)
        
        keystrokes, keystroke_encs: Tuple[Tensor, Tensor] = self.keystroke_network(actions)
        delays, delay_encs: Tuple[Tensor, Tensor] = self.delay_network(actions, keystroke_encs)
        mouse_positions: Tensor = self.mouse_network(actions, delay_encs, image_enc)

        return keystrokes, delays, mouse_positions
        
