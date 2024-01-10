import torch
from torch import nn, Tensor
from torch.nn import Module, TransformerDecoder, TransformerDecoderLayer

from typing import Optional


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
        num_layers=decoder_layer,
        norm=norm,
    )




class TransformerMemory(Module):
    USER_WARNING = False
    def __init__(
            self,
            input_feature_dims: int,
            output_feature_dims: int,
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

        # could probably use an MLP here instead for learning more sophisticated projections.
        self.linear_projection_in = nn.Linear(input_feature_dims, memory_feature_dims, bias=False)
        self.linear_projection_out = nn.Linear(memory_feature_dims, output_feature_dims, bias=False)

        self.initial_memory = nn.Parameter(torch.Rand((memory_size, memory_feature_dims)))
        self.positional_embeddings = nn.Parameter(torch.Rand((memory_size + 1, memory_feature_dims)))
        self.cls_token = nn.Parameter(torch.rand((1, memory_feature_dims)))

        self.memory = self.initial_memory

    def forward(self, feature_enc: Tensor) -> Tensor:
        if len(feature_enc.shape) == 3 and not TransformerMemory.USER_WARNING:
            print("User Warning: transformer memory has not had batches implemented! You may receive errors!")

        memory_seq = torch.cat((self.cls_token, self.memory), dim=1) + self.positional_embeddings
        feature_enc = self.linear_projection_in(feature_enc)

        memory_enc = self.decoder(memory_seq, feature_enc)[0]

        # note that this is treating the memory tensor like a stack [most recent -> least recent]
        self.memory[1:] = self.memory[:-1]
        self.memory[0] = memory_enc

        memory_enc = self.linear_projection_out(memory_enc)

        return memory_enc
