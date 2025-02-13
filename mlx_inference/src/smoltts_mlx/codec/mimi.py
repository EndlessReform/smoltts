from huggingface_hub import hf_hub_download
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pydantic import BaseModel
from typing import Any, List, Optional

from smoltts_mlx.codec.rvq import RVQConfig, MimiSplitResidualVectorQuantizer
from smoltts_mlx.codec.conv import (
    SeanetConfig,
    MimiConv1d,
    GroupedConvTranspose1d,
)
from smoltts_mlx.codec.seanet import MimiEncoder, MimiDecoder
from smoltts_mlx.codec.transformer import MimiTransformerConfig, MimiTransformer


class MimiConfig(BaseModel):
    seanet: SeanetConfig
    transformer: MimiTransformerConfig
    rvq: RVQConfig


def get_encodec_frame_rate(config: MimiConfig):
    hop_length = np.prod(config.seanet.ratios)
    return math.ceil(config.seanet.sampling_rate / hop_length)


class MimiModel(nn.Module):
    def __init__(self, config: MimiConfig):
        super().__init__()
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

    def _decode_frame(
        self, codes: mx.array, cache: Optional[List[Any]], is_step=False
    ) -> mx.array:
        embeddings = self.quantizer.decode(codes)
        embeddings = self.upsample(embeddings)
        decoder_outputs = self.decoder_transformer(embeddings, cache=cache)
        embeddings = decoder_outputs
        with mx.stream(mx.gpu):
            if is_step:
                outputs = self.decoder.step(embeddings)
            else:
                outputs = self.decoder(embeddings)
        out = mx.swapaxes(outputs, 1, 2)
        return out

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

    def decode_step(self, codes: mx.array, cache: Optional[List[Any]]) -> mx.array:
        audio_values = self._decode_frame(codes, cache, is_step=True)

        return audio_values


def load_mimi(format: str = "fp32") -> MimiModel:
    config = MimiConfig(
        seanet=SeanetConfig(), transformer=MimiTransformerConfig(), rvq=RVQConfig()
    )
    model_path = hf_hub_download("kyutai/mimi", "model.safetensors")
    model = MimiModel(config)
    state_dict = mx.load(model_path)

    # Yes, this is dumb.
    # The all-knowing maintainers of MLX decided to serialize conv1ds as NHWC instaed of NCHW,
    # despite the entire API surface being designed to mimic pytorch, because it's faster on apple silicon, and then
    # "helpfully" leaked that abstraction onto me.
    # But the convtrans1d is DIFFERENT yet again.
    def is_convtrans1d(key) -> bool:
        return (
            # Decoder only
            "decoder" in key
            and key.endswith(".conv.weight")
            and "block" not in key
            # Layer 0 is regular
            and "0" not in key
            # Final layer is regular
            and "14" not in key
        )

    def is_conv1d(key):
        return (
            key.endswith(".conv.weight")
            # RVQ proj
            or "quantizer.input_proj" in key
            or "quantizer.output_proj" in key
            and key != "upsample.conv.weight"
        )

    converted_dict = {
        k: v.transpose(1, 2, 0)
        if is_convtrans1d(k)
        else v.transpose(0, 2, 1)
        if is_conv1d(k)
        else v
        for k, v in state_dict.items()
    }
    dtype = mx.bfloat16 if format == "bf16" else mx.float32
    weight_list = [(k, v.astype(dtype)) for k, v in converted_dict.items()]

    model.load_weights(weight_list, strict=True)
    mx.eval(model.parameters())
    model.eval()

    return model
