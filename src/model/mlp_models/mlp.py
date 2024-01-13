import torch.functional as f
from torch import nn, Tensor
from torch.nn import Module
from typing import Tuple, List


def _fc_layer(input_dims: int, output_dims: int, activation: Module = nn.ReLU) -> Tuple[Module, Module]:
    return nn.Linear(input_dims, output_dims), activation()

def _mlp_layer(input_dims: int, hidden_dims: int, output_dims: int, depth: int = 2, activation: Module = nn.ReLU) -> List[Module]:
    mlp = []
    mlp.extend(_fc_layer(input_dims, hidden_dims, activation=activation))
    for _ in range(depth):
        mlp.extend(_fc_layer(hidden_dims, hidden_dims, activation=activation))
    mlp.extend(_fc_layer(hidden_dims, output_dims, activation=activation))

    return mlp

def _action_classifier_layer(input_dims: int, action_classes: int) -> List[Module]:
    return [
        nn.Linear(input_dims, action_classes),
        nn.Sigmoid(),
    ]

def _classifier_layer(input_dims: int, classes: int) -> List[Module]:
    return [
        nn.Linear(input_dims, classes),
        nn.Sigmoid(),
    ]


class KeystrokeClassifier(Module):
    def __init__(
            self,
            input_dims: int, 
            action_space: int,
            hidden_dims: int,
            encoder_dims: int,
    ) -> None:
        super(KeystrokeClassifier, self).__init__()
        
        self.encoder_dims = encoder_dims
        self.encoder = _mlp_layer(input_dims, hidden_dims, hidden_dims+encoder_dims)
        self.classifer = _action_classifier_layer(hidden_dims, action_space)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        encoding = self.encoder(x)
        return self.classifer(encoding[:-self.encoder_dims]), encoding[-self.encoder_dims:]
    
    
class BaseKeystrokeClassifier(KeystrokeClassifier):
    def __init__(
        self,
        input_dims: int,
        action_space: List[str],
        encoder_dims: int,
    ) -> None:
        super(BaseKeystrokeClassifier, self).__init__(
            input_dims,
            len(action_space),
            encoder_dims,
            1024,
        )


class DelayClassifier(Module):
    def __init__(
            self,
            input_dims: int, 
            delay_classes: int,
            hidden_dims: int,
            encoder_dims: int,
    ) -> None:
        super(DelayClassifier, self).__init__()
        
        self.encoder_dims = encoder_dims
        self.encoder = _mlp_layer(input_dims, hidden_dims, hidden_dims+encoder_dims)
        self.classifer = _classifier_layer(hidden_dims, delay_classes)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        encoding = self.encoder(x)
        return f.Sigmoid(self.classifer(encoding[:-self.encoder_dims])), encoding[-self.encoder_dims:]


class BaseDelayClassifier(DelayClassifier):
    def __init__(
        self,
        input_dims: int,
        delays: List[int],
        encoder_dims: int,
    ) -> None:
        super(BaseDelayClassifier, self).__init__(
            input_dims,
            len(delays),
            1024,
            encoder_dims,
        )
        