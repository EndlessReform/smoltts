from huggingface_hub import snapshot_download
from pathlib import Path
import mlx.core as mx
import time
from tokenizers import Tokenizer
from typing import Optional
from tqdm import tqdm

from smoltts_mlx.codec.mimi import load_mimi, MimiModel
from smoltts_mlx.lm.rq_transformer import (
    RQTransformerModelArgs,
)
from smoltts_mlx.lm.csm import CSMModel
from smoltts_mlx.lm.config import ModelType
from smoltts_mlx.lm.utils.prompt import CSMPromptEncoder
from smoltts_mlx.generate.csm import SingleBatchGenerator
from smoltts_mlx.generate.utils import GenerationSettings
from smoltts_mlx.io.wav import pcm_to_wav_bytes


class CSM:
    model: CSMModel
    config: RQTransformerModelArgs
    codec: MimiModel
    prompt_encoder: CSMPromptEncoder

    def __init__(
        self, model_id="jkeisling/csm-1b", checkpoint_dir: Optional[str] = None
    ):
        checkpoint_dir = Path(
            checkpoint_dir
            if checkpoint_dir is not None
            else snapshot_download(model_id)
        )
        config = RQTransformerModelArgs.from_json_file(
            str(checkpoint_dir / "config.json")
        )
        model_type = ModelType.csm_1b()

        config = RQTransformerModelArgs.from_json_file(
            str(checkpoint_dir / "config.json")
        )
        tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
        prompt_encoder = CSMPromptEncoder(tokenizer)

        model = CSMModel(config, model_type)
        model_path = str(checkpoint_dir / "model.safetensors")
        model.load_weights(model_path, strict=True)
        model = model.apply(lambda p: p.astype(mx.bfloat16))
        mx.eval(model.parameters())
        model.eval()
        self.codec = load_mimi()
        self.model = model
        self.prompt_encoder = prompt_encoder

    def __call__(self, text: str, speaker_id: int, temp: float = 0.9, top_k: int = 64):
        """
        Blocking E2E audio generation; returns 24khz PCM
        """
        prompt_segments = []
        mask_segments = []

        curr_tokens, curr_tokens_mask = self.prompt_encoder.tokenize_text(
            f"[{speaker_id}]{text}"
        )
        prompt_segments.append(curr_tokens)
        mask_segments.append(curr_tokens_mask)

        prompt = mx.stack(prompt_segments)
        prompt_mask = mx.stack(mask_segments)

        decode_start_time = time.time()
        # TODO pass through
        print(f"PROMPT SHAPE: {prompt.shape}")
        gen = SingleBatchGenerator(
            self.model,
            prompt,
            prompt_mask,
            GenerationSettings(default_temp=temp, top_k=top_k),
        )
        prefill_start_time = time.time()
        codes = [next(gen)]
        prefill_end_time = time.time()
        prefill_ms = (prefill_end_time - prefill_start_time) * 1000
        print(
            f"{prefill_ms:3f}ms prompt processing: {prompt.shape[-1]} tokens ({prompt.shape[-1] / (prefill_end_time - prefill_start_time):3f} tokens/s)"
        )

        # accumulate codes blocking
        for step in tqdm(gen):
            if step is not None:
                codes.append(step)

        mx.eval(codes)

        out_len = len(codes) - 1
        codes = mx.concat(codes, axis=-1)
        decode_end_time = time.time()
        decode_duration = decode_end_time - decode_start_time
        frame_rate = 12.5
        pcm = self.codec.decode(codes)
        print(
            f"Generated in {decode_duration:.2f}s ({(out_len / decode_duration):.2f} tokens/s, {((decode_duration * 1000) / out_len):.2f}ms/token), {(out_len / frame_rate) / decode_duration:.2f}x realtime"
        )
        return pcm
