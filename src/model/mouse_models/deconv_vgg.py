import torch
from torch import nn, Tensor
from torch.nn import Module

from typing import Optional, Tuple, List

from math import floor

from torch.nn.common_types import _size_2_t

class SpecialMaxUnpool2d(Module):
    def __init__(self, kernel_size: _size_2_t, stride: _size_2_t | None = None, padding: _size_2_t = 0) -> None:
        super().__init__()
        
        self.unpool = nn.MaxUnpool2d(kernel_size, stride, padding)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.unpool(x, torch.randint_like(x, 4).to(torch.int64))


def _deconv_layer(kernel_size: int, input_channels: int, output_channels: int, activation: Module = nn.ReLU, padding: int = 1) -> Tuple[Module, Module]:
    return nn.ConvTranspose2d(input_channels, output_channels, kernel_size, padding=padding), activation()


def _unpooling_layer() -> Module:
    return SpecialMaxUnpool2d(2, 2)

def _vgg_layer(kernel_size: int, feature_channels: int, input_channels: int, depth: int, activation: Module = nn.ReLU, padding: int = 1) -> List[Module]:
    layer = [_unpooling_layer()]
    for i in range(depth):
        if i == 0:
            layer.extend(_deconv_layer(kernel_size, input_channels, feature_channels, activation=activation, padding=padding))
        else:
            layer.extend(_deconv_layer(kernel_size, feature_channels, feature_channels, activation=activation, padding=padding))
    return layer
    

def _fc_layer(input_dims: int, output_dims: int, activation: Module = nn.ReLU) -> Tuple[Module, Module]:
    return nn.Linear(input_dims, output_dims), activation()

def _mlp_layer(input_dims: int, feature_dims: int, output_dims: int, depth: int = 2, activation: Module = nn.ReLU) -> Module:
    mlp = []
    mlp.extend(_fc_layer(input_dims, feature_dims, activation=activation))
    for _ in range(depth):
        mlp.extend(_fc_layer(feature_dims, feature_dims, activation=activation))
    mlp.extend(_fc_layer(feature_dims, output_dims, activation=activation))

    return nn.Sequential(*mlp)

def _compute_output_dims(*vgg_layers, output_dims: Tuple[int, int]) -> Tuple[int, int, int]:
    def in_dims(dim: int, stride: int, padding: int, kernel_size: int) -> int:
        # derived from https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html
        return floor((dim - kernel_size + 2*padding) / stride + 1)

    h, w = output_dims
    c = -1
    
    vgg_layers = list(vgg_layers)
    vgg_layers.reverse()
    
    for vgg_layer in vgg_layers:
        h = in_dims(dim=h, stride=2, padding=0, kernel_size=2)
        w = in_dims(dim=w, stride=2, padding=0, kernel_size=2)
        c = vgg_layer[2]
    
    assert isinstance(h, int) or isinstance(w, int), f"deconv vgg received in valid dims! output should be multiple of {2**len(vgg_layers)}"

    return c, h, w


class DeconvVGG(Module):
    """
    DeconvVGG network as similar to https://arxiv.org/pdf/1505.04366.pdf. Softmax layer added at the end to convert into probabilities of mouse position
    """
    def __init__(self, *vgg_layers: Tuple[int, int, int, int], input_dims: int = 2048, mlp_feature_dims: int = 2048, mlp_output_dims: int = 4096, mlp_depth = 2, output_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG, self).__init__()

        self.model = []

        mlp_output_dims = _compute_output_dims(*vgg_layers, output_dims=output_dims)

        self.model.append(_mlp_layer(input_dims, mlp_feature_dims, mlp_output_dims[0]*mlp_output_dims[1]*mlp_output_dims[2]))
        self.model.append(nn.Unflatten(-1, mlp_output_dims))

        for vgg_layer in vgg_layers:
            self.model.extend(_vgg_layer(*vgg_layer))

        self.model.append(nn.Sigmoid())

        self.model = nn.Sequential(*self.model)
        self.forward = self.model
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def get_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class Mini(DeconvVGG):
    """
    Mini DeconvVGG that is cpu friendly
    """
    # tuples are organised as (kernel_size: int, feature_channels: int, input_channels: int, depth: int)
    CONFIG = [
        (3, 1, 8, 1),
        (3, 8, 16, 2),
        (3, 16, 32, 2),
        (3, 32, 64, 2),
        (3, 64, 64, 2), 
    ]
    CONFIG.reverse()
    def __init__(self, input_dims: int = 256, mlp_feature_dims: int = 256, mlp_output_dims: int = 256, mlp_depth = 2, output_dims: Tuple[int, int] = (400, 400)):
        super(Mini, self).__init__(
            *Mini.CONFIG,
            mlp_feature_dims=mlp_feature_dims,
            mlp_output_dims=mlp_output_dims,
            mlp_depth=mlp_depth,
            input_dims=input_dims,
            output_dims=output_dims,
        )

class DeconvVGG11(DeconvVGG):
    # tuples are organised as (kernel_size, feature_channels, input_channels, depth)
    CONFIG = [
        (3, 1, 64, 1),
        (3, 64, 128, 1),
        (3, 128, 256, 2),
        (3, 256, 512, 2),
        (3, 512, 512, 2),
    ].reverse()
    def __init__(self, input_dims: int = 4096, mlp_feature_dims: int = 4096, mlp_output_dims: int = 4096, mlp_depth = 2, output_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG11, self).__init__(
            *DeconvVGG11.CONFIG,
            mlp_feature_dims=mlp_feature_dims,
            mlp_output_dims=mlp_output_dims,
            mlp_depth=mlp_depth,
            input_dims=input_dims,
            output_dims=output_dims,
        )


class DeconvVGG13(DeconvVGG):
    # tuples are organised as (kernel_size, feature_channels, input_channels, depth)
    CONFIG = [
        (3, 1, 64, 2),
        (3, 64, 128, 2),
        (3, 128, 256, 2),
        (3, 256, 512, 2),
        (3, 512, 512, 2),
    ].reverse()
    def __init__(self, input_dims: int = 4096, mlp_feature_dims: int = 4096, mlp_output_dims: int = 4096, mlp_depth = 2, output_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG13, self).__init__(
            *DeconvVGG13.CONFIG,
            mlp_feature_dims=mlp_feature_dims,
            mlp_output_dims=mlp_output_dims,
            mlp_depth=mlp_depth,
            input_dims=input_dims,
            output_dims=output_dims,
        )


class DeconvVGG16(DeconvVGG):
    # tuples are organised as (kernel_size, feature_channels, input_channels, depth)
    CONFIG = [
        (3, 1, 64, 2),
        (3, 64, 128, 2),
        (3, 128, 256, 3),
        (3, 256, 512, 3),
        (3, 512, 512, 3),
    ].reverse()
    def __init__(self, input_dims: int = 4096, mlp_feature_dims: int = 4096, mlp_output_dims: int = 4096, mlp_depth = 2, output_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG16, self).__init__(
            *DeconvVGG16.CONFIG,
            mlp_feature_dims=mlp_feature_dims,
            mlp_output_dims=mlp_output_dims,
            mlp_depth=mlp_depth,
            input_dims=input_dims,
            output_dims=output_dims,
        )


class DeconvVGG19(DeconvVGG):
    # tuples are organised as (kernel_size: int, feature_channels: int, input_channels: int, depth: int)
    CONFIG = [
        (3, 1, 64, 2),
        (3, 64, 128, 2),
        (3, 128, 256, 4),
        (3, 256, 512, 4),
        (3, 512, 512, 4),
    ]
    CONFIG.reverse()
    def __init__(self, input_dims: int = 4096, mlp_feature_dims: int = 4096, mlp_output_dims: int = 4096, mlp_depth = 2, output_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG19, self).__init__(
            *DeconvVGG19.CONFIG,
            mlp_feature_dims=mlp_feature_dims,
            mlp_output_dims=mlp_output_dims,
            mlp_depth=mlp_depth,
            input_dims=input_dims,
            output_dims=output_dims,
        )
