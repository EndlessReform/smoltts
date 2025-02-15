from huggingface_hub import snapshot_download
import mlx.core as mx
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
from typing import List, Optional

from smoltts_mlx.codec.mimi import load_mimi
from smoltts_mlx.lm.config import ModelType
from smoltts_mlx.lm.utils.prompt import PromptEncoder
from smoltts_mlx.lm.rq_transformer import (
    RQTransformer,
    RQTransformerModelArgs,
    TokenConfig,
)
from smoltts_mlx.lm.cache import make_prompt_cache
from smoltts_mlx.lm.generate import (
    generate_blocking,
    SingleBatchGenerator,
    GenerationSettings,
)


class SmolTTS:
    def __init__(
        self,
        model_id="jkeisling/smoltts_v0",
        checkpoint_dir: Optional[str] = None,
    ):
        checkpoint_dir = Path(
            checkpoint_dir
            if checkpoint_dir is not None
            else snapshot_download(model_id)
        )
        config = RQTransformerModelArgs.from_json_file(
            str(checkpoint_dir / "config.json")
        )
        # TODO support other configs once changes are made
        model_type = ModelType.smoltts_v0()

        tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
        token_config = TokenConfig.from_tokenizer(
            model=model_type, tokenizer=tokenizer, config=config
        )

        model = RQTransformer(config, token_config, model_type)
        model_path = str(checkpoint_dir / "model.safetensors")
        model.load_weights(model_path, strict=True)
        mx.eval(model.parameters())
        model.eval()

        prompt_encoder = PromptEncoder.from_model(tokenizer, model)
        codec = load_mimi()

        self.lm = model
        self.prompt_encoder = prompt_encoder
        self.codec = codec

        # TODO load speakers here
        # TODO make this configurable
        self.sampling_rate = 24_000

    def __call__(
        self,
        input: str,
        voice: Optional[str] = "heart",
        speaker: Optional[mx.array] = None,
    ) -> np.ndarray:
        """
        Returns flattened PCM array
        """
        prompt = self._get_prompt(
            input, voice if voice is not None else "heart", sysprompt=speaker
        )
        # TODO make this configurable
        gen = generate_blocking(self.lm, prompt, GenerationSettings())
        out = self.codec.decode(gen)
        mx.metal.clear_cache()

        return np.array(out).flatten()

    def stream(self, input: str, voice: Optional[str] = "heart"):
        prompt = self._get_prompt(input, voice if voice is not None else "0")
        frame_gen = SingleBatchGenerator(self.lm, prompt, GenerationSettings())
        mimi_cache = make_prompt_cache(self.codec.decoder_transformer)

        for frame in tqdm(frame_gen):
            audio_tokens = frame.vq_tensor[:, 1:, :]
            pcm_chunk = self.codec.decode_step(audio_tokens, mimi_cache)
            audio_data = np.array(pcm_chunk).flatten()
            yield audio_data

        self.codec.decoder.reset()
        mx.metal.clear_cache()

    def create_speaker(
        self, samples: List[dict], system_prompt: Optional[str] = None
    ) -> mx.array:
        turns = []
        for sample in samples:
            if "audio" not in sample or "text" not in sample:
                raise ValueError(
                    f"Sample must contain both 'text' and 'audio' but got {sample.keys()}"
                )
            user_prompt = self.prompt_encoder.encode_text_turn("user", sample["text"])
            encoded_audio = self.codec.encode(mx.array(sample["audio"]))
            codes = self.prompt_encoder.encode_vq(encoded_audio.squeeze(0)[:8, :])
            turns.append(user_prompt)
            turns.append(codes)

        if system_prompt is not None:
            turns = [
                self.prompt_encoder.encode_text_turn("system", system_prompt),
                *turns,
            ]

        return mx.concat(turns, axis=1)

    def _get_prompt(self, input: str, voice: str, sysprompt=None):
        # TODO remove this after voices are configurable
        voice_map = {
            k: v
            for v, k in enumerate(
                [
                    "heart",
                    "bella",
                    "nova",
                    "sky",
                    "sarah",
                    "michael",
                    "fenrir",
                    "liam",
                    "emma",
                    "isabella",
                    "fable",
                ]
            )
        }
        voice_id = voice_map.get(voice, 0)

        if sysprompt is None:
            sysprompt = self.prompt_encoder.encode_text_turn(
                "system", f"<|speaker:{voice_id}|>"
            )
        user_prompt = self.prompt_encoder.encode_text_turn("user", input)
        assistant_prefix = self.prompt_encoder.encode_text_turn("assistant")
        prompt = mx.concat([sysprompt, user_prompt, assistant_prefix], axis=1)[
            mx.newaxis, :, :
        ]
        return prompt
