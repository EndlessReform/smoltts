import mlx.core as mx
from pathlib import Path
import time
from tokenizers import Tokenizer

from mlx_inference.model.dual_ar import DualARModelArgs, DualARTransformer, TokenConfig
from mlx_inference.model.config import ModelType
from mlx_inference.model.cache import make_prompt_cache


def main():
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

    # Initialize cache
    cache = make_prompt_cache(model)

    # Initial prefill with prompt
    prefill_start_time = time.time()
    prompt = mx.zeros([1, 9, 32], dtype=mx.uint32)
    logits, hidden_states = model.forward_generate(prompt, cache=cache)
    mx.eval(logits, hidden_states)
    prefill_end_time = time.time()
    print(
        f"CPU prefill with cache in {((prefill_end_time - prefill_start_time) * 1000):.3f}ms"
    )

    # Get argmax token
    x = hidden_states[mx.newaxis, :, :]
    fast_cache = make_prompt_cache(model, is_fast=True)
    for codebook_idx in range(0, model.config.num_codebooks):
        fast_logits = model.forward_generate_fast(x, cache=fast_cache)
        mx.eval(fast_logits)
        # mx.save(
        #     f"zeroes_code{codebook_idx + 1}_mlx.npy", fast_logits.astype(mx.float32)
        # )
        next_token = mx.argmax(fast_logits, axis=-1)
        print(next_token)

        print(f"generated {codebook_idx}")
        x = model.fast_embeddings(next_token)


if __name__ == "__main__":
    main()
