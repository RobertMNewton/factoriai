import torch
from torch import nn, Tensor
from torch.nn import Module, TransformerDecoder, TransformerDecoderLayer

from typing import Optional, Tuple


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

def _mlp_layer(input_dims: int, feature_dims: int, output_dims: int, depth: int = 2, activation: Module = nn.ReLU) -> Module:
    mlp = []
    mlp.extend(_fc_layer(input_dims, feature_dims, activation=activation))
    for _ in range(depth):
        mlp.extend(_fc_layer(feature_dims, feature_dims, activation=activation))
    mlp.extend(_fc_layer(feature_dims, output_dims, activation=activation))

    return nn.Sequential(*mlp)

class TransformerMemory(Module):
    USER_WARNING = False
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
            projection_feature_dims: int,
            memory_feature_dims: int,
            memory_size: int = 256,
            decoder_depth: int = 12,
            decoder_norm: Optional[Module] = None,
            attn_heads: int = 8,
            mlp_dim: int = 2048
    ) -> None:
        super(TransformerMemory, self).__init__()

        self.decoder = _transformer_decoder(
            feature_dim=memory_feature_dims,
            attn_heads=attn_heads,
            depth=decoder_depth,
            mlp_dim=mlp_dim,
            norm=decoder_norm,
        )

        # note sure if MLPs for the linear projection is overkill but this should help with feature extraction
        # for high dimensional discrepancy between input, memory and output feature dims
        self.linear_projection_in = _mlp_layer(input_feature_dims, projection_feature_dims, memory_feature_dims)
        self.linear_projection_out = _mlp_layer(memory_feature_dims, projection_feature_dims, output_feature_dims)

        self.initial_memory = nn.Parameter(torch.rand((memory_size, memory_feature_dims)), requires_grad=True)
        self.positional_embeddings = nn.Parameter(torch.rand((memory_size + 1, memory_feature_dims)), requires_grad=True)
        self.cls_token = nn.Parameter(torch.rand((1, memory_feature_dims)), requires_grad=True)

        self.memory = torch.clone(self.initial_memory)

    def forward(self, feature_enc: Tensor) -> Tensor:
        if len(feature_enc.shape) == 3 and not TransformerMemory.USER_WARNING:
            print("User Warning: transformer memory has not had batches implemented! You may receive errors!")

        memory_seq = torch.cat((self.cls_token, self.memory), dim=0) + self.positional_embeddings
        feature_enc = self.linear_projection_in(feature_enc)

        memory_enc = self.decoder(memory_seq, feature_enc)[0]

        # note that this is treating the memory tensor like a stack [most recent -> least recent]
        self.memory[1:] = torch.clone(self.memory[:-1])
        self.memory[0] = memory_enc

        memory_enc = self.linear_projection_out(memory_enc)

        return memory_enc
    
    def get_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Mini(TransformerMemory):
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
            memory_size: int = 256,
    ) -> None:
        super(Mini, self).__init__(
            input_feature_dims=input_feature_dims,
            output_feature_dims=output_feature_dims,
            projection_feature_dims=64,
            memory_size=memory_size,
            memory_feature_dims=64,
            decoder_depth=6,
            attn_heads=8,
            mlp_dim=256,
            decoder_norm=nn.LayerNorm(64),
        )

class Small(TransformerMemory):
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
            memory_size: int = 256,
    ) -> None:
        super(Small, self).__init__(
            input_feature_dims=input_feature_dims,
            output_feature_dims=output_feature_dims,
            projection_feature_dims=256,
            memory_size=memory_size,
            memory_feature_dims=256,
            decoder_depth=6,
            attn_heads=8,
            mlp_dim=512,
            decoder_norm=nn.LayerNorm(256),
        )

class Medium(TransformerMemory):
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
            memory_size: int = 256,
    ) -> None:
        super(Medium, self).__init__(
            input_feature_dims=input_feature_dims,
            output_feature_dims=output_feature_dims,
            projection_feature_dims=512,
            memory_size=memory_size,
            memory_feature_dims=512,
            decoder_depth=12,
            attn_heads=16,
            mlp_dim=1024,
            decoder_norm=nn.LayerNorm(512),
        )

class Base(TransformerMemory):
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
            memory_size: int = 256,
    ) -> None:
        super(Base, self).__init__(
            input_feature_dims=input_feature_dims,
            output_feature_dims=output_feature_dims,
            projection_feature_dims=1024,
            memory_size=memory_size,
            memory_feature_dims=1024,
            decoder_depth=24,
            attn_heads=32,
            mlp_dim=2048,
            decoder_norm=nn.LayerNorm(1024),
        )

class Large(TransformerMemory):
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
            memory_size: int = 256,
    ) -> None:
        super(Large, self).__init__(
            input_feature_dims=input_feature_dims,
            output_feature_dims=output_feature_dims,
            projection_feature_dims=2048,
            memory_size=memory_size,
            memory_feature_dims=2048,
            decoder_depth=48,
            attn_heads=32,
            mlp_dim=4096,
            decoder_norm=nn.LayerNorm(2048),
        )

class XLarge(TransformerMemory):
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
            memory_size=memory_size,
            memory_feature_dims=2048,
            decoder_depth=96,
            attn_heads=32,
            mlp_dim=4096,
            decoder_norm=nn.LayerNorm(2048),
        )
        