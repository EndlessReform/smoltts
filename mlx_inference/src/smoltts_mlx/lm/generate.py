import mlx.core as mx
from pydantic import BaseModel, Field
import time
from tqdm import tqdm
from typing import Any, Optional, List

from smoltts_mlx.lm.rq_transformer import RQTransformer
from smoltts_mlx.lm.cache import make_prompt_cache, KVCache
from smoltts_mlx.lm.utils.samplers import min_p_sampling


class GenerationSettings(BaseModel):
    default_temp: float = Field(default=0.7)
    default_fast_temp: Optional[float] = Field(default=0.7)
    min_p: Optional[float] = Field(default=None)
    max_new_tokens: int = Field(default=1024)


class VQToken(BaseModel):
    semantic_code: int
    audio_codes: Optional[Any]
    vq_tensor: Any


class SingleBatchGenerator:
    model: RQTransformer
    input_pos: int
    generation_settings: GenerationSettings
    prompt: Optional[mx.array]
    previous_codes: Optional[List[int]]
    audio_only: bool
    cache: List[KVCache]

    def __init__(
        self,
        model: RQTransformer,
        prompt: mx.array,
        generation_settings: GenerationSettings,
        audio_only: bool = True,
    ):
        self.model = model
        # TODO handle KV cache
        self.input_pos = 0
        # TODO handle this
        self.max_new_tokens = (
            generation_settings.max_new_tokens
            if generation_settings.max_new_tokens is not None
            else model.config.max_seq_len
        )
        self.generation_settings = generation_settings
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
        if self.generation_settings.default_temp == 0.0:
            token_ids = mx.argmax(slow_logits, axis=-1)
        elif self.generation_settings.min_p is not None:
            token_ids = min_p_sampling(
                slow_logits,
                min_p=self.generation_settings.min_p,
                temperature=self.generation_settings.default_temp,
            )
        else:
            # TODO improve sampling, I just want SOME output
            slow_logits = slow_logits / self.generation_settings.default_temp
            token_ids = mx.random.categorical(slow_logits)

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
        for i in range(0, self.model.max_fast_seqlen):
            fast_logits = self.model.forward_generate_fast(x, i, cache=fast_cache)
            mx.eval(fast_logits)

            # TODO handle sampling, esp. if it sounds terrible
            if (
                fast_temp := self.generation_settings.default_fast_temp
            ) is not None and fast_temp > 0:
                if self.generation_settings.min_p is not None:
                    next_token_tensor = min_p_sampling(
                        fast_logits.squeeze(0),
                        min_p=self.generation_settings.min_p,
                        temperature=self.generation_settings.default_fast_temp,
                    )
                    next_token_tensor = next_token_tensor[mx.newaxis, :]
                else:
                    fast_logits = fast_logits / fast_temp
                    next_token_tensor = mx.random.categorical(fast_logits)
            else:
                next_token_tensor = mx.argmax(fast_logits, axis=-1)

            # model GETS higher
            code = next_token_tensor.flatten()[0]
            if self.model.config.depthwise_wte:
                offset = i if self.model.config.duplicate_code_0 else i + 1
                next_token_tensor += max(0, offset * self.model.config.codebook_size)

            x = self.model.fast_embeddings(next_token_tensor)
            codes.append(code)

        codes_tensor = mx.array([slow_token_id, *codes], dtype=mx.uint32)[
            mx.newaxis, :, mx.newaxis
        ]
        if (
            slow_token_id >= self.model.token_config.semantic_start_id
            and self.model.token_config.semantic_end_id is not None
            and slow_token_id <= self.model.token_config.semantic_end_id
        ):
            audio_code = slow_token_id - self.model.token_config.semantic_start_id
            codes_arr = (
                codes if self.model.config.duplicate_code_0 else [audio_code, *codes]
            )
            audio_tensor = mx.array(codes_arr, dtype=mx.uint32)[
                mx.newaxis, :, mx.newaxis
            ]
        else:
            audio_tensor = None

        self.input_pos += prompt_length if self.input_pos is None else 1
        self.prompt = (
            None
            if self.audio_only and slow_token_id == self.model.token_config.im_end_id
            else codes_tensor
        )
        return VQToken(
            semantic_code=slow_token_id.tolist(),
            audio_codes=audio_tensor,
            vq_tensor=codes_tensor,
        )


def generate_blocking(
    model: RQTransformer,
    prompt: mx.array,
    generation_settings: GenerationSettings,
    audio_only: bool = True,
) -> mx.array:
    prompt_size = prompt.shape[-1]
    token_generator = SingleBatchGenerator(
        model,
        prompt,
        generation_settings,
        audio_only,
    )
    prefill_start_time = time.time()
    first_vq_token = next(token_generator)
    prefill_end_time = time.time()
    prefill_ms = (prefill_end_time - prefill_start_time) * 1000
    print(
        f"{prefill_ms:3f}ms prompt processing: {prompt_size} tokens ({prompt_size / (prefill_end_time - prefill_start_time):3f} tokens/s)"
    )

    previous_vq_codes = (
        [first_vq_token.audio_codes] if audio_only else [first_vq_token.vq_tensor]
    )

    decode_start_time = time.time()
    for maybe_vq_token in tqdm(token_generator):
        if audio_only:
            if maybe_vq_token.audio_codes is not None:
                previous_vq_codes.append(maybe_vq_token.audio_codes)
        else:
            previous_vq_codes.append(maybe_vq_token.vq_tensor)
    decode_end_time = time.time()
    decode_duration = decode_end_time - decode_start_time

    out_tokens = mx.concat(previous_vq_codes, axis=-1)
    out_len = len(previous_vq_codes) - 1
    frame_rate = 12.5 if model.model_type.family == "dual_ar" else 21.535
    print(
        f"Generated in {decode_duration:.2f}s ({(out_len / decode_duration):.2f} tokens/s, {((decode_duration * 1000) / out_len):.2f}ms/token), {(out_len / frame_rate) / decode_duration:.2f}x realtime"
    )
    mx.eval(out_tokens)
    return out_tokens
