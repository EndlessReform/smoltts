import argparse
import mlx.core as mx
import numpy as np
from pathlib import Path
import time
from tokenizers import Tokenizer
from tqdm import tqdm

from smoltts_mlx.lm.rq_transformer import (
    RQTransformerModelArgs,
)
from smoltts_mlx.lm.csm import CSMModel
from smoltts_mlx.lm.config import ModelType
from smoltts_mlx.lm.utils.prompt import CSMPromptEncoder
from smoltts_mlx.generate.csm import SingleBatchGenerator
from smoltts_mlx.generate.utils import GenerationSettings
from smoltts_mlx.lm.cache import make_prompt_cache
from smoltts_mlx.codec.mimi import load_mimi
from smoltts_mlx.io.wav import pcm_to_wav_bytes

parser = argparse.ArgumentParser(
    description="A simple one-off CLI generator for DualAR models"
)
parser.add_argument("--text", type=str, default="Hello world!")
parser.add_argument("--speaker", type=int, default=0)
parser.add_argument("--checkpoint", type=str, default="./inits/csm_1b")


def main():
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint)
    model_type = ModelType.csm_1b()

    load_start_time = time.time()
    config = RQTransformerModelArgs.from_json_file(str(checkpoint_dir / "config.json"))
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    prompt_encoder = CSMPromptEncoder(tokenizer)

    model = CSMModel(config, model_type)
    model_path = str(checkpoint_dir / "model.safetensors")
    model.load_weights(model_path, strict=True)
    model = model.apply(lambda p: p.astype(mx.bfloat16))
    mx.eval(model.parameters())
    model.eval()
    mimi_model = load_mimi()
    load_end_time = time.time()

    print(f"Loaded model and config in {load_end_time - load_start_time:.3f} seconds")

    text = "[0]okay good! im happy enough with this at least its working and yes i know! I feel i take the lazy approach which turns to be the most painful approach"
    curr_tokens, curr_tokens_mask = prompt_encoder.tokenize_text(text)
    curr_tokens = curr_tokens[mx.newaxis, :, :]
    curr_tokens_mask = curr_tokens_mask[mx.newaxis, :, :]

    decode_start_time = time.time()
    gen = SingleBatchGenerator(
        model, curr_tokens, curr_tokens_mask, GenerationSettings(max_new_tokens=256)
    )
    codes = []
    for step in tqdm(gen):
        if step is not None:
            codes.append(step)

    mx.eval(codes)
    out_len = len(codes) - 1
    codes = mx.concat(codes, axis=-1)
    decode_end_time = time.time()
    decode_duration = decode_end_time - decode_start_time
    frame_rate = 12.5
    pcm = mimi_model.decode(codes)
    print(
        f"Generated in {decode_duration:.2f}s ({(out_len / decode_duration):.2f} tokens/s, {((decode_duration * 1000) / out_len):.2f}ms/token), {(out_len / frame_rate) / decode_duration:.2f}x realtime"
    )

    print(codes.shape)
    with open("out.wav", "wb") as f:
        f.write(pcm_to_wav_bytes(np.array(pcm)))

    raise ValueError("DONE")


if __name__ == "__main__":
    main()
