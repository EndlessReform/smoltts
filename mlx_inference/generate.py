import mlx.core as mx
from pathlib import Path
import time
from tokenizers import Tokenizer

from mlx_inference.model.dual_ar import DualARModelArgs, DualARTransformer, TokenConfig
from mlx_inference.model.config import ModelType


def main():
    # TODO totally arbitrary and terrible,
    # just testing numerical accuracy for what I know
    checkpoint_dir = Path("./inits/smoltts_byte_kokoro")
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
    mx.eval(model.parameters())
    model.eval()
    load_end_time = time.time()
    print(f"Loaded model and config in {load_end_time - load_start_time:.3f} seconds")

    # prompt = mx.zeros([1, 8, 32])
    print("TODO: verify forward pass")
    exit(0)


if __name__ == "__main__":
    main()
