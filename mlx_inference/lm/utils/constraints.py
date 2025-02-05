import mlx.core as mx
from typing import List

from mlx_inference.lm.config import ModelType
from mlx_inference.lm.rq_transformer import TokenConfig


def constrain_logits_to_audio(
    x: mx.array, model_type: ModelType, token_config: TokenConfig
) -> mx.array:
    if model_type.family == "dual_ar" or model_type.version == "1.5":
        # Base layer uses <|semantic:n|> range up top
        if token_config.im_end_id == token_config.semantic_start_id - 1:
            # Saves us an indexop
            return x[:, :, token_config.im_end_id :]
        else:
            im_end_prob = x[:, :, token_config.im_end_id]
            semantic_token_range = x[:, :, token_config.semantic_start_id :]
            # TODO this is probably wrong
            return mx.concat([im_end_prob, semantic_token_range], -1)
    else:
        return x


def rescale_semantic_tokens(
    tokens: List[int], model_type: ModelType, token_config: TokenConfig
):
    token_range_is_contiguous = (
        token_config.im_end_id == token_config.semantic_start_id - 1
    )

    def rescale_token(token: int) -> int:
        if token_range_is_contiguous:
            return token + token_config.im_end_id
        elif token == 0:
            return token_config.im_end_id
        else:
            return token - 1 + token_config.semantic_start_id

    if model_type.family == "dual_ar" or model_type.version == "1.5":
        return [rescale_token(t) for t in tokens]
    else:
        return tokens
