import uvicorn
import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI
import huggingface_hub
import mlx.core as mx
from pathlib import Path
import rustymimi
import time
from tokenizers import Tokenizer

from mlx_inference.tts_core import TTSCore
from mlx_inference.routes import openai
from mlx_inference.lm.rq_transformer import (
    RQTransformerModelArgs,
    RQTransformer,
    TokenConfig,
)
from mlx_inference.lm.utils.prompt import PromptEncoder
import mlx_inference
import mlx_inference.settings


def get_mimi_path():
    """Get Mimi tokenizer weights from Hugging Face."""
    repo_id = "kyutai/moshiko-mlx-bf16"
    filename = "tokenizer-e351c8d8-checkpoint125.safetensors"
    return huggingface_hub.hf_hub_download(repo_id, filename)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file")


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_dir = Path(settings.checkpoint_dir)
    model_type = settings.model_type

    load_start_time = time.time()
    print("Loading model configuration and tokenizer...")
    config = RQTransformerModelArgs.from_json_file(str(checkpoint_dir / "config.json"))
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    token_config = TokenConfig.from_tokenizer(
        model=model_type, tokenizer=tokenizer, config=config
    )

    print("Loading model...")
    model = RQTransformer(config, token_config, model_type)
    model_path = str(checkpoint_dir / "model.safetensors")
    model.load_weights(model_path, strict=True)
    mx.eval(model.parameters())
    model.eval()

    print("Downloading and initializing Mimi tokenizer...")
    mimi_path = get_mimi_path()
    mimi_tokenizer = rustymimi.Tokenizer(mimi_path)
    prompt_encoder = PromptEncoder.from_model(tokenizer, model)

    app.state.tts_core = TTSCore(
        model=model,
        tokenizer=tokenizer,
        mimi_tokenizer=mimi_tokenizer,
        prompt_encoder=prompt_encoder,
        settings=settings,
    )

    load_end_time = time.time()
    print(f"Loaded model and config in {load_end_time - load_start_time:.3f} seconds")
    print(f"Default device: {mx.default_device()}")

    yield
    print("shutting down")


app = FastAPI(lifespan=lifespan)
app.include_router(openai.router)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    settings = mlx_inference.settings.ServerSettings.get_settings(args.config)
    app.state.settings = settings

    uvicorn.run(app, host="0.0.0.0", port=8000)
