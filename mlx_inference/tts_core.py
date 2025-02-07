import mlx.core as mx
import numpy as np
import time
from typing import Union

from mlx_inference.lm.generate import generate_blocking
from mlx_inference.io.wav import pcm_to_wav_bytes


class TTSCore:
    def __init__(self, model, tokenizer, mimi_tokenizer, prompt_encoder, settings):
        self.model = model
        self.tokenizer = tokenizer
        self.mimi_tokenizer = mimi_tokenizer
        self.prompt_encoder = prompt_encoder
        self.settings = settings

    def resolve_speaker_id(self, voice: Union[str, int]) -> int:
        # TODO: Fix speaker cache
        if isinstance(voice, int):
            return voice
        return 0

    def generate_audio(
        self, input_text: str, voice: Union[str, int], response_format: str = "wav"
    ):
        speaker_id = self.resolve_speaker_id(voice)

        sysprompt = self.prompt_encoder.encode_text_turn(
            "system", f"<|speaker:{speaker_id}|>"
        )
        user_prompt = self.prompt_encoder.encode_text_turn("user", input_text)
        assistant_prefix = self.prompt_encoder.encode_text_turn("assistant")
        prompt = mx.concat([sysprompt, user_prompt, assistant_prefix], axis=1)[
            mx.newaxis, :, :
        ]

        # Generate semantic tokens
        gen = generate_blocking(
            self.model, prompt, self.settings.generation, audio_only=True
        )

        # Convert to numpy and decode
        tokens = np.array(gen).astype(np.uint32)
        start_time = time.time()
        pcm_data = self.mimi_tokenizer.decode(tokens)
        end_time = time.time()
        print(f"Took {end_time - start_time:.2f}s to decode")

        return pcm_to_wav_bytes(pcm_data)
