import mlx.core as mx
from typing import Optional, List

from smoltts_mlx.lm.cache import make_prompt_cache, KVCache
from smoltts_mlx.lm.csm import CSMModel
from smoltts_mlx.lm.utils.samplers import min_p_sampling, top_k_sampling
from smoltts_mlx.generate.utils import GenerationSettings


class SingleBatchGenerator:
    model: CSMModel
    n_generated: int
    max_new_tokens: int
    generation_settings: GenerationSettings
    prompt: Optional[mx.array]
    prompt_mask: Optional[mx.array]
    audio_only: bool
    cache: List[KVCache]
    previous_codes: Optional[mx.array]

    def __init__(
        self,
        model: CSMModel,
        prompt: mx.array,
        prompt_mask: mx.array,
        generation_settings: GenerationSettings,
    ):
        self.model = model
        # TODO handle
        self.n_generated = 0
        self.max_new_tokens = (
            generation_settings.max_new_tokens
            if generation_settings.max_new_tokens is not None
            else model.config.max_seq_len
        )

        self.prompt = prompt
        self.prompt_mask = prompt_mask
        self.cache = make_prompt_cache(model)
        self.previous_codes = None
        self.generation_settings = generation_settings

    def __iter__(self):
        return self

    def __next__(self):
        if self.n_generated > self.max_new_tokens:
            raise StopIteration
        elif self.prompt is None or self.prompt_mask is None:
            raise StopIteration

        code0_logits, hidden_states = self.model.forward_generate(
            self.prompt, self.prompt_mask, self.cache
        )
        mx.eval(code0_logits, hidden_states)

        # TODO more rigorous sampling
        token_ids = min_p_sampling(code0_logits, min_p=0.1, temperature=1)
        # token_ids = mx.argmax(code0_logits, keepdims=True)
        c0_sample = token_ids[mx.newaxis, :]
        c0_embed = self.model.embed_audio(0, c0_sample)
        curr_h = mx.concat([hidden_states[:, mx.newaxis, :], c0_embed], axis=1)
        curr_sample = c0_sample
        mx.save("curr_h_mlx", curr_h)

        decoder_cache = make_prompt_cache(self.model, is_fast=True)
        for i in range(1, self.model.config.num_codebooks):
            code_logits = self.model.forward_generate_fast(
                curr_h, decoder_cache, codebook_idx=i
            )
            # if i == 1:
            #     mx.save("first_residual_code_mlx", code_logits)
            # if i == 2:
            #     mx.save("second_residual_code_mlx", code_logits)
            #     raise ValueError("FUCK")
            # # TODO
            # ci_sample = mx.argmax(code_logits, keepdims=True)
            ci_sample = top_k_sampling(code_logits, top_k=64, temperature=0.95)[
                mx.newaxis, :
            ]
            curr_h = self.model.embed_audio(i, ci_sample)
            curr_sample = mx.concat([curr_sample, ci_sample], axis=1)

        if mx.all(curr_sample == 0):
            # Gen is over
            self.prompt = None
            self.prompt_mask = None
            return None

        # Bookkeeping for next round
        self.prompt = mx.concat(
            [curr_sample, mx.zeros([1, 1], dtype=mx.int64)], axis=1
        )[:, mx.newaxis, :]
        # god help me i just need this to be done
        audio_mask = mx.ones_like(curr_sample) == 1
        text_mask = mx.zeros([1, 1]) != 0

        self.prompt_mask = mx.concat([audio_mask, text_mask], axis=1)[:, mx.newaxis, :]
        self.n_generated += 1
        return curr_sample[:, :, mx.newaxis]
