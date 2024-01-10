import torch
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

def _mlp_layer(input_dims: int, feature_dims: int, output_dims: int, depth: int = 2, activation: Module = nn.ReLU) -> List[Module]:
    return [_fc_layer(input_dims, feature_dims, activation=activation)] \
        + [_fc_layer(feature_dims, feature_dims, activation=activation) for _ in range(depth)] \
        + [_fc_layer(feature_dims, output_dims, activation=activation)]

def _compute_output_dims(*vgg_layers, input_dims: Tuple[int, int, int]) -> Tuple[int, int, int]:
    def out_dims(in_dims: int, padding: int, dilation: int, kernel_size: int, stride: int) -> int:
        # taken from https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        return floor((in_dims + 2*padding - dilation*kernel_size - 1)/stride + 1)

    h, w, c = input_dims

    for vgg_layer in vgg_layer:
        # the dims are only reduced by the max pooling in a VGG due to the padding in the deconvolutions
        w = out_dims(in_dims=w, padding=1, dilation=1, kernel_size=2, stride=2)
        h = out_dims(in_dims=h, padding=1, dilation=1, kernel_size=2, stride=2)
        c = vgg_layer[2]

    return h, w, c


class DeconvVGG(Module):
    """
    DeconvVGG network as described in https://arxiv.org/pdf/1505.04366.pdf
    """
    def __init__(self, *vgg_layers: Tuple[int, int, int, int], mlp_feature_dims: int = 4096, mlp_output_dims: int = 4096, mlp_depth = 2, input_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG, self).__init__()

        self.model = []

        mlp_output_dims = _compute_output_dims(vgg_layers, input_dims)

        self.model.append(_mlp_layer(mlp_input_dims, mlp_feature_dims, mlp_output_dims))
        self.model.append(nn.Unflatten(-1, mlp_output_dims))

        for vgg_layer in vgg_layers:
            self.model.append(*_vgg_layer(*vgg_layer))
        
        mlp_input_dims = _compute_output_dims(vgg_layers, input_dims)
        mlp_input_dims = mlp_input_dims[0] * mlp_input_dims[1]

        self.model.append(nn.Flatten())
        self.model.append(_mlp_layer(mlp_input_dims, mlp_feature_dims, mlp_output_dims))

        self.model = nn.Sequential(*self.model)
        self.forward = self.model


class DeconvVGG11(DeconvVGG):
    # tuples are organised as (kernel_size, input_channels, feature_channels, depth)
    CONFIG = [
        (3, 3, 64, 1),
        (3, 64, 128, 1),
        (3, 128, 256, 2),
        (3, 256, 512, 2),
        (3, 512, 512, 2),
    ]
    def __init__(self, mlp_feature_dims: int = 4096, mlp_output_dims: int = 4096, mlp_depth = 2, input_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG11, self).__init__(
            *DeconvVGG11.CONFIG,
            mlp_feature_dims=mlp_feature_dims,
            mlp_output_dims=mlp_output_dims,
            mlp_depth=mlp_depth,
            input_dims=input_dims,
        )


class DeconvVGG13(DeconvVGG):
    # tuples are organised as (kernel_size, input_channels, feature_channels, depth)
    CONFIG = [
        (3, 3, 64, 2),
        (3, 64, 128, 2),
        (3, 128, 256, 2),
        (3, 256, 512, 2),
        (3, 512, 512, 2),
    ]
    def __init__(self, mlp_feature_dims: int = 4096, mlp_output_dims: int = 4096, mlp_depth = 2, input_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG13, self).__init__(
            *DeconvVGG13.CONFIG,
            mlp_feature_dims=mlp_feature_dims,
            mlp_output_dims=mlp_output_dims,
            mlp_depth=mlp_depth,
            input_dims=input_dims,
        )


class DeconvVGG16(DeconvVGG):
    # tuples are organised as (kernel_size, input_channels, feature_channels, depth)
    CONFIG = [
        (3, 3, 64, 2),
        (3, 64, 128, 2),
        (3, 128, 256, 3),
        (3, 256, 512, 3),
        (3, 512, 512, 3),
    ]
    def __init__(self, mlp_feature_dims: int = 4096, mlp_output_dims: int = 4096, mlp_depth = 2, input_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG16, self).__init__(
            *DeconvVGG16.CONFIG,
            mlp_feature_dims=mlp_feature_dims,
            mlp_output_dims=mlp_output_dims,
            mlp_depth=mlp_depth,
            input_dims=input_dims,
        )


class DeconvVGG19(DeconvVGG):
    # tuples are organised as (kernel_size, input_channels, feature_channels, depth)
    CONFIG = [
        (3, 3, 64, 2),
        (3, 64, 128, 2),
        (3, 128, 256, 4),
        (3, 256, 512, 4),
        (3, 512, 512, 4),
    ]
    def __init__(self, mlp_feature_dims: int = 4096, mlp_output_dims: int = 4096, mlp_depth = 2, input_dims: Tuple[int, int] = (400, 400)):
        super(DeconvVGG19, self).__init__(
            *DeconvVGG19.CONFIG,
            mlp_feature_dims=mlp_feature_dims,
            mlp_output_dims=mlp_output_dims,
            mlp_depth=mlp_depth,
            input_dims=input_dims,
        )
