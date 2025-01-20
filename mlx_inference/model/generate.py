import mlx.core as mx
from pydantic import BaseModel
import time
from tqdm import tqdm
from typing import Any, Optional, List

from mlx_inference.model.dual_ar import DualARTransformer
from mlx_inference.model.cache import make_prompt_cache, KVCache
from mlx_inference.model.utils.constraints import (
    constrain_logits_to_audio,
    rescale_semantic_tokens,
)


class VQToken(BaseModel):
    semantic_code: int
    vq_codes: List[int]
    vq_tensor: Any


class SingleBatchGenerator:
    model: DualARTransformer
    input_pos: int
    max_new_tokens: int
    prompt: Optional[mx.array]
    previous_codes: Optional[List[int]]
    audio_only: bool
    cache: List[KVCache]

    def __init__(
        self,
        model: DualARTransformer,
        prompt: mx.array,
        audio_only: bool = True,
        max_new_tokens: Optional[int] = None,
    ):
        self.model = model
        # TODO handle KV cache
        self.input_pos = 0
        # TODO handle this
        self.max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else model.config.max_seq_len
        )
        self.audio_only = audio_only
        self.prompt = prompt
        self.previous_codes = None
        self.slow_cache = make_prompt_cache(model)

    def __iter__(self):
        return self

    def __next__(self):
        if self.input_pos > self.max_new_tokens:
            raise StopIteration
        elif self.prompt is None:
            # Previous iteration told us to stop
            raise StopIteration

        x = self.prompt
        prompt_length = x.shape[-1]

        x = x if x.ndim == 3 else x[mx.newaxis, :, :]

        logits, hidden_states = self.model.forward_generate(
            self.prompt, self.slow_cache
        )
        mx.eval(logits, hidden_states)
        logits = logits if logits.ndim == 3 else logits[mx.newaxis, :, :]
        # slow_logits = (
        #     constrain_logits_to_audio(
        #         logits, self.model.model_type, self.model.token_config
        #     )
        #     if (
        #         is_modern := self.audio_only
        #         and self.model.model_type.family == "dual_ar"
        #         or self.model.model_type.version == "1.5"
        #     )
        #     else logits
        # )
        slow_logits = logits
        # TODO handle sampling, I just want SOME output
        token_ids = mx.argmax(slow_logits, axis=-1)
        # slow_token_id = (
        #     rescale_semantic_tokens(
        #         token_ids, self.model.model_type, self.model.token_config
        #     )[0]
        #     if is_modern
        #     else token_ids[0]
        # )
        slow_token_id = token_ids.flatten()[0]

        codes = []
        x = hidden_states[mx.newaxis, :, :]
        fast_cache = make_prompt_cache(self.model, is_fast=True)
        for _ in range(0, self.model.config.num_codebooks):
            fast_logits = self.model.forward_generate_fast(x, cache=fast_cache)
            mx.eval(fast_logits)

            # TODO handle sampling, esp. if it sounds terrible
            next_token_tensor = mx.argmax(fast_logits, axis=-1)
            x = self.model.fast_embeddings(next_token_tensor)
            codes.append(next_token_tensor.flatten()[0])

        codes_tensor = mx.array([slow_token_id, *codes], dtype=mx.uint32)[
            mx.newaxis, :, mx.newaxis
        ]
        self.input_pos += prompt_length if self.input_pos is None else 1
        self.prompt = (
            None
            if self.audio_only and slow_token_id == self.model.token_config.im_end_id
            else codes_tensor
        )
        return VQToken(
            semantic_code=slow_token_id.tolist(), vq_codes=codes, vq_tensor=codes_tensor
        )


def generate_blocking(
    model: DualARTransformer,
    prompt: mx.array,
    audio_only: bool = True,
    max_new_tokens: Optional[int] = None,
) -> mx.array:
    prompt_size = prompt.shape[-1]
    token_generator = SingleBatchGenerator(model, prompt, audio_only, max_new_tokens)
    prefill_start_time = time.time()
    first_vq_token = next(token_generator)
    prefill_end_time = time.time()
    prefill_ms = (prefill_end_time - prefill_start_time) * 1000
    print(
        f"{prefill_ms:3f}ms prompt processing: {prompt_size} tokens ({prompt_size / (prefill_end_time - prefill_start_time):3f} tokens/s)"
    )

    previous_vq_codes = [first_vq_token.vq_tensor]

    decode_start_time = time.time()
    for maybe_vq_token in tqdm(token_generator):
        previous_vq_codes.append(maybe_vq_token.vq_tensor)
    decode_end_time = time.time()
    decode_duration = decode_end_time - decode_start_time

    full_output = mx.concat(previous_vq_codes, axis=-1)
    out_tokens = full_output[:, 1:, :] if audio_only else full_output
    out_len = len(previous_vq_codes) - 1
    frame_rate = 12.5 if model.model_type.family == "dual_ar" else 21.535
    print(
        f"Generated in {decode_duration:.2f}s ({(out_len / decode_duration):.2f} tokens/s, {((decode_duration * 1000) / out_len):.2f}ms/token), RTF: {(out_len / frame_rate) / decode_duration:.2f}"
    )
    mx.eval(out_tokens)
    return out_tokens
