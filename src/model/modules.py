import torch
from torch.nn import Module
from torch import Tensor
from typing import Tuple, Union, List, Any, Dict


def _get_shape(ts: Union[Tensor, Tuple[Tensor]]) -> List[Tuple[int]]:
    if isinstance(ts, Tensor):
        return [ts.shape]
    else:
        return [t.shape for t in ts]


def _validate_shape(ts: Union[Tensor, Tuple[Tensor]], shape: List[Tuple[int]]) -> bool:
    for actual_shape, expected_shape in zip(_get_shape(ts), shape):
        if actual_shape != expected_shape:
            return False
    return True


class NetworkWrapper(Module):
    def __init__(self, input_dims: List[Tuple[int]], output_dims: List[Tuple[int]], network: Module = None):
        super(VisionModule, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims

        self.network = network
        
    def set_network(self, network: Module) -> None:
        self.network = network

    def forward(self, x: Union[Tensor, Tuple[Tensor]], **kwargs) -> Union[Tensor, Tuple[Tensor]]:
        assert _validate_shape(x, self.input_dims), \
            f"{self.__class__} encountered input with invalid dims. Got {_get_shape(x)}, expected {self.input_dims}"
        assert self.network is not None, \
            f"{self.__class__} has not had its network set"
        
        x = self.network(x, **kwargs)
        
        assert _validate_shape(x, self.output_dims), \
            f"{self.__class__} encountered input with invalid dims. Got {_get_shape(x)}, expected {self.output_dims}"
        
        return x
    
    def decode(self, x: Union[Tensor, Tuple[Tensor]]) -> Any:
        raise f"{self.__class__} cannot have its output decoded!"
    
    def get_size(self) -> int:
        return sum(param.numel() for param in self.parameters)
    

class VisionModule(NetworkWrapper):
    """
    Network wrapper for vision network of architecture. Performs validation of input and output.
    """
    

class MemoryModule(NetworkWrapper):
    """
    Wraps memory networks
    """
    
    
class ActionModule(NetworkWrapper):
    """
    Wraps action networks
    """
    
    
class DelayModule(NetworkWrapper):
    """
    Wraps delay classification networks
    """
    def __init__(self, input_dims: List[Tuple[int]], delays: List[int], encoding_size: int, network: Module = None):
        self.delay_map = {i:x for i, x in enumerate(delays)}
        super().__init__(input_dims, [(-1, len(self.delay_map)), (-1, encoding_size)], network)
        
    def decode(self, x: Tensor | Tuple[Tensor]) -> Any:
        _, classification = torch.max(x, dim=-1)
        return self.delay_map[classification]


class KeystrokeModule(NetworkWrapper):
    """
    Wraps keystroke classification networks
    """
    def __init__(self, input_dims: List[Tuple[int]], keys: List[str], encoding_size: int, network: Module = None):
        self.key_map = {i: s for i, s in enumerate(keys)}
        super().__init__(input_dims, [(-1, len(self.key_map), 2), (-1, encoding_size)], network)
        
    def decode(self, x: Tensor | Tuple[Tensor]) -> Any:
        _, classifications = torch.max(x, dim=-1)
        return [self.key_map[classification] for classification in classifications if classification == 1]
    

class MouseModule(NetworkWrapper):
    """
    Wraps mouse classification networks
    """
    def decode(self, x: Tensor | Tuple[Tensor]) -> Any:
        ys, x = torch.max(x, dim=-1)
        _, y = torch.max(ys)
        
        return x[y], y
