from torch.nn import Module
from torch import Tensor
from typing import Tuple

class VisionModule(Module):
    """
    Network wrapper for vision network of architecture. Performs validation of input and output.
    """
    def __init__(self, input_dims: Tuple[int, int, int], output_dims: Tuple[int], network: Module):
        super(VisionModule, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims

        self.network = network

    def forward(self, image: Tensor) -> Tensor:
        if len(image.shape) == 3:
            assert image.shape == self.input_dims,  \
                f"vision module received invalid input dimensions: expected {self.input_dims} got {image.shape}"
        elif len(image.shape) == 4:
            assert image.shape[-3:] == self.input_dims,  \
                f"vision module received invalid input dimensions: expected {self.input_dims} got {image.shape}"
        else:
            raise ValueError(f"vision modules received image with invalid dims: expect 3 or 4 got {len(image.shape)}")

        res = self.network(image)

        if len(res.shape) == 1:
            assert res.shape == self.input_dims,  \
                f"vision module received invalid output dimensions: expected {self.output_dims} got {res.shape}"
        elif len(res.shape) == 2:
            assert res.shape[-3:] == self.input_dims,  \
                f"vision module received invalid output dimensions: expected {self.output_dims} got {res.shape}"
        else:
            raise ValueError(f"vision modules received invalid network output with invalid dims: expect 1 or 2 got {len(res.shape)}")
        
        return res

