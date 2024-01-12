from torch import nn
from torch.nn import Module

from typing import Tuple, List

from math import floor


def _deconv_layer(kernel_size: int, input_channels: int, output_channels: int, activation: Module = nn.ReLU, padding: int = 1) -> Tuple[Module, Module]:
    return nn.ConvTranspose2d(input_channels, output_channels, kernel_size, padding=padding), activation()


def _unpooling_layer() -> Module:
    return nn.MaxUnpool2d(2, 2)

def _vgg_layer(kernel_size: int, feature_channels: int, input_channels: int, depth: int, activation: Module = nn.ReLU, padding: int = 1) -> List[Module]:
    return [_unpooling_layer()] \
    + [_deconv_layer(kernel_size, input_channels, feature_channels, activation=activation, padding=padding) for _ in range(depth)]

def _fc_layer(input_dims: int, output_dims: int, activation: Module = nn.ReLU) -> Tuple[Module, Module]:
    return nn.Linear(input_dims, output_dims), activation()

def _mlp_layer(input_dims: int, feature_dims: int, output_dims: int, depth: int = 2, activation: Module = nn.ReLU) -> Module:
    mlp = []
    mlp.extend(_fc_layer(input_dims, feature_dims, activation=activation))
    for _ in range(depth):
        mlp.extend(_fc_layer(feature_dims, feature_dims, activation=activation))
    mlp.extend(_fc_layer(feature_dims, output_dims, activation=activation))

    return nn.Sequential(*mlp)

def _compute_output_dims(*vgg_layers, output_dims: Tuple[int, int, int]) -> Tuple[int, int, int]:
    def in_dims(dim: int, stride: int, padding: int, kernel_size: int) -> int:
        # taken from https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html
        return (dim + 1)*stride - 2*padding + kernel_size

    h, w, c = output_dims
    for vgg_layer in [*vgg_layers].reverse():
        h = in_dims(dim=h, stride=2, padding=0, kernel_size=2)
        w = in_dims(dim=w, stride=2, padding=0, kernel_size=2)
        c = vgg_layer[2]

    return c, h, w


class DeconvVGG(Module):
    """
    DeconvVGG network as described in https://arxiv.org/pdf/1505.04366.pdf. Softmax layer added at the end to convert into probabilities of mouse position
    """
    def __init__(self, *vgg_layers: Tuple[int, int, int, int], input_dims: int = 4096, mlp_feature_dims: int = 4096, mlp_output_dims: int = 4096, mlp_depth = 2, output_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG, self).__init__()

        self.model = []

        mlp_output_dims = _compute_output_dims(vgg_layers, input_dims)

        self.model.append(_mlp_layer(input_dims, mlp_feature_dims, mlp_output_dims))
        self.model.append(nn.Unflatten(-1, mlp_output_dims))

        for vgg_layer in vgg_layers:
            self.model.append(*_vgg_layer(*vgg_layer))

        self.model.append(nn.Sigmoid())

        self.model = nn.Sequential(*self.model)
        self.forward = self.model
    
    def get_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
    # tuples are organised as (kernel_size, feature_channels, input_channels, depth)
    CONFIG = [
        (3, 1, 64, 2),
        (3, 64, 128, 2),
        (3, 128, 256, 4),
        (3, 256, 512, 4),
        (3, 512, 512, 4),
    ].reverse()
    def __init__(self, input_dims: int = 4096, mlp_feature_dims: int = 4096, mlp_output_dims: int = 4096, mlp_depth = 2, output_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG19, self).__init__(
            *DeconvVGG19.CONFIG,
            mlp_feature_dims=mlp_feature_dims,
            mlp_output_dims=mlp_output_dims,
            mlp_depth=mlp_depth,
            input_dims=input_dims,
            output_dims=output_dims,
        )
