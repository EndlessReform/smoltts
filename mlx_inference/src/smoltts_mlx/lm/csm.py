import mlx.core as mx
import mlx.nn as nn

from smoltts_mlx.lm.rq_transformer import RQTransformerModelArgs, TransformerBlock
from smoltts_mlx.lm.config import ModelType


class CSM(nn.Module):
    def __init__(self, config: RQTransformerModelArgs, model_type: ModelType):
        super().__init__()
        if model_type.family != "csm":
            raise ValueError("Cannot load weights")

        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks, config.dim
        )
        self.codebook0_head = nn.Linear(config.dim, config.codebook_size, bias=False)
        # TODO handle this, this sucks
        self.audio_head = mx.zeros(
            shape=[config.num_codebooks - 1, config.fast_dim, config.codebook_size]
        )

        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]

        self.fast_project_in = nn.Linear(config.dim, config.fast_dim, bias=False)
        self.fast_layers = [
            TransformerBlock(config, is_fast=True) for _ in range(config.n_fast_layer)
        ]
        self.fast_norm = nn.RMSNorm(config.fast_dim, eps=config.norm_eps)
