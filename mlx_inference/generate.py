import mlx.core as mx
from pathlib import Path
import time
from tokenizers import Tokenizer

from mlx_inference.model.dual_ar import DualARModelArgs, DualARTransformer, TokenConfig
from mlx_inference.model.config import ModelType


def main():
    # TODO totally arbitrary and terrible,
    # just testing numerical accuracy for what I know
    checkpoint_dir = Path("./inits/smoltts_byte_reference")
    model_type = ModelType(family="dual_ar", version=None, codec="mimi")

    load_start_time = time.time()
    config = DualARModelArgs.from_json_file(str(checkpoint_dir / "config.json"))
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    token_config = TokenConfig.from_tokenizer(
        model=model_type, tokenizer=tokenizer, config=config
    )

    model = DualARTransformer(config, token_config, model_type)
    model_path = str(checkpoint_dir / "model.safetensors")
    model.load_weights(model_path, strict=True)
    model = model.apply(lambda p: p.astype(mx.float32))
    mx.eval(model.parameters())
    model.eval()
    load_end_time = time.time()
    print(f"Loaded model and config in {load_end_time - load_start_time:.3f} seconds")

    prefill_start_time = time.time()
    prompt = mx.zeros([1, 9, 32], dtype=mx.uint32)
    logits, hidden_states = model.forward_generate(prompt)
    mx.eval(logits, hidden_states)
    prefill_end_time = time.time()
    print("TODO: verify forward pass accuracy")
    print(
        f"Naive CPU prefill in {((prefill_end_time - prefill_start_time) * 1000):.3f}ms"
    )
    mx.save("zeroes_logits_mlx.npy", logits.astype(mx.float32)[mx.newaxis, :, :])
    mx.save("zeroes_hidden_mlx.npy", hidden_states.astype(mx.float32)[mx.newaxis, :, :])
    # prefill_start_time = time.time()
    # prompt2 = mx.ones([1, 9, 32], dtype=mx.uint32)
    # mx.eval(model.forward_generate(prompt2))
    # prefill_end_time = time.time()
    # print(
    #     f"after warmup CPU prefill in {((prefill_end_time - prefill_start_time) * 1000):.3f}ms"
    # )

    exit(0)


if __name__ == "__main__":
    main()
