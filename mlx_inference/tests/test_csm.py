import argparse
import mlx.core as mx
from pathlib import Path
import time
from tokenizers import Tokenizer

from smoltts_mlx.lm.rq_transformer import (
    RQTransformerModelArgs,
)
from smoltts_mlx.lm.csm import CSM
from smoltts_mlx.lm.config import ModelType
from smoltts_mlx.lm.generate import SingleBatchGenerator

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

    model = CSM(config, model_type)
    model_path = str(checkpoint_dir / "model.safetensors")
    model.load_weights(model_path, strict=True)
    # model = model.apply(lambda p: p.astype(mx.float32))
    mx.eval(model.parameters())
    model.eval()
    load_end_time = time.time()
    print(f"Loaded model and config in {load_end_time - load_start_time:.3f} seconds")
    raise ValueError("TODO")

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
