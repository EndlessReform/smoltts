import argparse
import mlx.core as mx
from pathlib import Path
import time
from tokenizers import Tokenizer

from smoltts_mlx.lm.rq_transformer import (
    RQTransformerModelArgs,
)
from smoltts_mlx.lm.csm import CSMModel
from smoltts_mlx.lm.config import ModelType
from smoltts_mlx.lm.utils.prompt import CSMPromptEncoder
from smoltts_mlx.lm.cache import make_prompt_cache

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
    model = model.apply(lambda p: p.astype(mx.float32))
    mx.eval(model.parameters())
    model.eval()
    load_end_time = time.time()

    print(f"Loaded model and config in {load_end_time - load_start_time:.3f} seconds")

    text = "[0]okay good! im happy enough with this at least its working and yes i know! I feel i take the lazy approach which turns to be the most painful approach"
    curr_tokens, curr_tokens_mask = prompt_encoder.tokenize_text(text)

    curr_tokens = curr_tokens[mx.newaxis, :, :]
    curr_tokens_mask = curr_tokens_mask[mx.newaxis, :, :]

    (code0_logits, x) = model.forward_generate(curr_tokens, curr_tokens_mask, None)

    print("Generated backbone")
    # TODO add sampling
    c0_sample = mx.argmax(code0_logits, keepdims=True)
    c0_embed = model.embed_audio(0, c0_sample)

    curr_h = mx.concat([x[:, mx.newaxis, :], c0_embed], axis=1)
    mx.eval(c0_sample)
    curr_sample = c0_sample

    curr_pos = mx.repeat(
        mx.arange(0, curr_h.shape[1])[mx.newaxis, :], curr_h.shape[0], 1
    )

    decoder_cache = make_prompt_cache(model, is_fast=True)
    for i in range(1, model.config.num_codebooks):
        code_logits = model.forward_generate_fast(curr_h, decoder_cache, codebook_idx=i)

        ci_sample = mx.argmax(code_logits, keepdims=True)

        curr_h = model.embed_audio(i, ci_sample)
        curr_sample = mx.concat([curr_sample, ci_sample], axis=1)
        curr_pos = curr_pos[:, -1:] + 1

    final = curr_sample
    print(final)
    print("Generated one round")

    # prompt_encoder = PromptEncoder.from_model(tokenizer, model)
    # sysprompt = prompt_encoder.encode_text_turn("system", f"<|speaker:{args.speaker}|>")
    # user_prompt = prompt_encoder.encode_text_turn("user", args.text)
    # assistant_prefix = prompt_encoder.encode_text_turn("assistant")
    # print([p.shape for p in [sysprompt, user_prompt, assistant_prefix]])
    # prompt = mx.concat([sysprompt, user_prompt, assistant_prefix], axis=1)[
    #     mx.newaxis, :, :
    # ]


if __name__ == "__main__":
    main()
