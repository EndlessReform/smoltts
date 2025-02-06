import argparse
import mlx.core as mx
from pathlib import Path
import time
from tokenizers import Tokenizer

from mlx_inference.lm.rq_transformer import (
    RQTransformerModelArgs,
    RQTransformer,
    TokenConfig,
)
from mlx_inference.lm.config import ModelType
from mlx_inference.lm.generate import SingleBatchGenerator
from mlx_inference.lm.utils.prompt import PromptEncoder

parser = argparse.ArgumentParser(
    description="A simple one-off CLI generator for DualAR models"
)
parser.add_argument("--text", type=str, default="Hello world!")
parser.add_argument("--speaker", type=int, default=0)
parser.add_argument("--checkpoint", type=str, default="./inits/smoltts_byte_reference")


def main():
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint)
    model_type = ModelType(family="dual_ar", version=None, codec="mimi")

    load_start_time = time.time()
    config = RQTransformerModelArgs.from_json_file(str(checkpoint_dir / "config.json"))
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    token_config = TokenConfig.from_tokenizer(
        model=model_type, tokenizer=tokenizer, config=config
    )

    model = RQTransformer(config, token_config, model_type)
    model_path = str(checkpoint_dir / "model.safetensors")
    model.load_weights(model_path, strict=True)
    # model = model.apply(lambda p: p.astype(mx.float32))
    mx.eval(model.parameters())
    model.eval()
    load_end_time = time.time()
    print(f"Loaded model and config in {load_end_time - load_start_time:.3f} seconds")

    # Initialize cache
    prompt = mx.zeros([1, 9, 32], mx.uint32)
    trace_file = "mlx_trace.gputrace"
    mx.metal.start_capture(trace_file)
    # prompt_encoder = PromptEncoder.from_model(tokenizer, model)
    # sysprompt = prompt_encoder.encode_text_turn("system", f"<|speaker:{args.speaker}|>")
    # user_prompt = prompt_encoder.encode_text_turn("user", args.text)
    # assistant_prefix = prompt_encoder.encode_text_turn("assistant")
    # print([p.shape for p in [sysprompt, user_prompt, assistant_prefix]])
    # prompt = mx.concat([sysprompt, user_prompt, assistant_prefix], axis=1)[
    #     mx.newaxis, :, :
    # ]
    generator = SingleBatchGenerator(model, prompt, audio_only=True)
    next(generator)
    next(generator)

    mx.metal.stop_capture()


if __name__ == "__main__":
    main()
