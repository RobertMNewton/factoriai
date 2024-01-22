import torch
from torch import nn, Tensor
from torch.nn import Module, TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as f

from typing import Tuple, Optional


class BreakLoop(Exception):
    pass


def _transformer_decoder(feature_dim: int, attn_heads: int, depth: int = 12, mlp_dim: int = 2048, norm: Optional[Module] = None) -> TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=feature_dim,
        nhead=attn_heads,
        dim_feedforward=mlp_dim,
        batch_first=True,
        norm_first=True,
    )

    return TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=depth,
        norm=norm,
    )

def _fc_layer(input_dims: int, output_dims: int, activation: Module = nn.ReLU) -> Tuple[Module, Module]:
    return nn.Linear(input_dims, output_dims), activation()

def _mlp_layer(input_dims: int, feature_dims: int, output_dims: int, depth: int = 2, activation: Module = nn.ReLU, classify: bool = False) -> Module:
    mlp = []
    mlp.extend(_fc_layer(input_dims, feature_dims, activation=activation))
    for _ in range(depth):
        mlp.extend(_fc_layer(feature_dims, feature_dims, activation=activation))
    mlp.extend(_fc_layer(feature_dims, output_dims, activation=activation))
    
    if classify:
        mlp.extend(nn.Sigmoid())

    return nn.Sequential(*mlp)


class ActionTransformer(Module):
    def __init__(
        self,
        input_feature_dim: int,
        hidden_feature_dim: int,
        output_feature_dim: int,
        depth: int,
        attn_heads: int,
        mlp_dim: int,
        ) -> None:
        super(ActionTransformer, self).__init__()
        
        self.decoder = _transformer_decoder(
            feature_dim=hidden_feature_dim,
            attn_heads=attn_heads,
            depth=depth,
            mlp_dim=mlp_dim,
            norm=nn.LayerNorm(hidden_feature_dim),
        )        
        
        self.linear_projection_in = _mlp_layer(input_feature_dim, int((input_feature_dim + hidden_feature_dim)/2), hidden_feature_dim)
        self.linear_projection_out = _mlp_layer(hidden_feature_dim, int((output_feature_dim + 2 + hidden_feature_dim)/2), output_feature_dim + 2)
        
        self.start_token = nn.Parameter(torch.rand((1, hidden_feature_dim)), requires_grad=True)
        
    def forward(self, feature: Tensor, train: Optional[int] = None, max_tokens: int = 20) -> Tuple[Tensor, Tensor]:
        """
        Action Transformer forward function. Train is an optional integer representing number of actions to sample. If 
        left as None, then will sample until reaches max_tokens or end_token classifier indicates to.
        
        Returns action encodings (including end token) and end token classifications for all action encodings
        """
        feature_encoding: Tensor = self.linear_projection_in(feature)
        
        if train is not None:
            max_tokens = train
        
        actions = torch.clone(self.start_token)    
        try:
            for _ in range(max_tokens):
                
                actions = torch.cat((actions, self.decoder(actions, actions)[-1].unsqueeze(0)), dim=0)
                
                if train is None:
                    probs = f.sigmoid(self.linear_projection_out(actions)[-1, -2:])
                    if probs[1] > probs[0]:
                        raise BreakLoop
        except BreakLoop:
            pass
        except Exception as e:
            raise e
        
        actions = self.linear_projection_out(actions)
        
        return actions[1:, :-2], f.sigmoid(actions[1:, -2:])
        
    
class Mini(ActionTransformer):
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
    ) -> None:
        super(Mini, self).__init__(
            input_feature_dim=input_feature_dims,
            hidden_feature_dim=64,
            output_feature_dim=output_feature_dims,
            depth=6,
            attn_heads=8,
            mlp_dim=256,
        )

class Small(ActionTransformer):
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
            memory_size: int = 256,
    ) -> None:
        super(Small, self).__init__(
            input_feature_dims,
            output_feature_dims,
            memory_size,
            6,
            8,
            512,
        )

class Medium(ActionTransformer):
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
    ) -> None:
        super(Medium, self).__init__(
            input_feature_dims=input_feature_dims,
            output_feature_dims=output_feature_dims,
            projection_feature_dims=512,
            hidden_feature_dim=512,
            depth=12,
            attn_heads=16,
            mlp_dim=1024,
            decoder_norm=nn.LayerNorm(512),
        )

class Base(ActionTransformer):
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
    ) -> None:
        super(Base, self).__init__(
            input_feature_dims=input_feature_dims,
            output_feature_dims=output_feature_dims,
            projection_feature_dims=1024,
            hidden_feature_dim=1024,
            depth=24,
            attn_heads=32,
            mlp_dim=2048,
            decoder_norm=nn.LayerNorm(1024),
        )

class Large(ActionTransformer):
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
    ) -> None:
        super(Large, self).__init__(
            input_feature_dims=input_feature_dims,
            output_feature_dims=output_feature_dims,
            projection_feature_dims=2048,
            hidden_feature_dim=2048,
            depth=48,
            attn_heads=32,
            mlp_dim=4096,
            decoder_norm=nn.LayerNorm(2048),
        )

class XLarge(ActionTransformer):
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
            memory_size: int = 256,
    ) -> None:
        super(XLarge, self).__init__(
            input_feature_dims=input_feature_dims,
            output_feature_dims=output_feature_dims,
            projection_feature_dims=2048,
            hidden_feature_dim=2048,
            depth=96,
            attn_heads=32,
            mlp_dim=4096,
            decoder_norm=nn.LayerNorm(2048),
        )    