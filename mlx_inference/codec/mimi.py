import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pydantic import BaseModel
from typing import Any, List, Optional

from mlx_inference.codec.rvq import RVQConfig, MimiSplitResidualVectorQuantizer
from mlx_inference.codec.conv import (
    SeanetConfig,
    MimiConv1d,
    GroupedConvTranspose1d,
)
from mlx_inference.codec.seanet import MimiEncoder, MimiDecoder
from mlx_inference.codec.transformer import MimiTransformerConfig, MimiTransformer


class MimiConfig(BaseModel):
    seanet: SeanetConfig
    transformer: MimiTransformerConfig
    rvq: RVQConfig


def get_encodec_frame_rate(config: MimiConfig):
    hop_length = np.prod(config.seanet.ratios)
    return math.ceil(config.seanet.sampling_rate / hop_length)


class MimiModel(nn.Module):
    def __init__(self, config: MimiConfig):
        super().__init__()  # Add this line
        self.config = config

        self.encoder = MimiEncoder(config.seanet)
        self.encoder_transformer = MimiTransformer(config.transformer)
        encodec_frame_rate = get_encodec_frame_rate(config)

        self.downsample = MimiConv1d(
            config.seanet,
            config.seanet.dimension,
            config.seanet.dimension,
            kernel_size=2 * int(encodec_frame_rate / config.rvq.frame_rate),
            stride=2,
            bias=False,
            pad_mode="replicate",
        )
        kernel_size = 2 * int(encodec_frame_rate / config.rvq.frame_rate)
        self.upsample = GroupedConvTranspose1d(
            config.seanet,
            config.seanet.dimension,
            config.seanet.dimension,
            kernel_size=kernel_size,
            stride=2,
            bias=False,
            groups=512,
        )

        self.decoder_transformer = MimiTransformer(config.transformer)
        self.decoder = MimiDecoder(config.seanet)

        self.quantizer = MimiSplitResidualVectorQuantizer(config.rvq)

    def _decode_frame(self, codes: mx.array, cache: Optional[List[Any]]) -> mx.array:
        embeddings = self.quantizer.decode(codes)
        print(f"Quantizer decode done; shape: {embeddings.shape}")
        mx.save("dequantized_embeddings_mlx.npy", embeddings)
        embeddings = self.upsample(embeddings)
        print(f"Upsample done; shape: {embeddings.shape}")
        mx.save("upsample_mlx.npy", mx.swapaxes(embeddings, 1, 2))
        decoder_outputs = self.decoder_transformer(embeddings, cache=cache)
        # embeddings = decoder_outputs[0].transpose(1, 2)
        # embeddings = decoder_outputs[0]
        embeddings = decoder_outputs
        print(f"Transformer done; shape: {embeddings.shape}")
        outputs = self.decoder(embeddings)
        return mx.swapaxes(outputs, 1, 2)

    def decode(
        self,
        audio_codes: mx.array,
        cache: Optional[List[Any]] = None,
        padding_mask: Optional[mx.array] = None,
    ):
        audio_values = self._decode_frame(audio_codes, cache)

        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[:, :, : padding_mask.shape[-1]]

        return audio_values
