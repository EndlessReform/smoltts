import uvicorn
import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
import mlx.core as mx

import time
from tokenizers import Tokenizer

from smoltts_mlx.codec.mimi import load_mimi
from smoltts_mlx.lm.rq_transformer import (
    RQTransformerModelArgs,
    RQTransformer,
    TokenConfig,
)
from smoltts_mlx.lm.utils.prompt import PromptEncoder
from smoltts_mlx.server.tts_core import TTSCore
from smoltts_mlx.server.routes import openai, elevenlabs
from smoltts_mlx.server.settings import ServerSettings


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    checkpoint_dir = settings.get_checkpoint_dir()
    model_type = settings.model_type

    load_start_time = time.time()
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
    mimi_tokenizer = load_mimi()
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

    yield
    print("shutting down")


app = FastAPI(lifespan=lifespan)
app.include_router(openai.router)
app.include_router(elevenlabs.router)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--port", type=int, help="Port to run on on (default: 8000)")
    args = parser.parse_args()

    settings = ServerSettings.get_settings(args.config)
    app.state.settings = settings

    port = args.port if args.port is not None else 8000

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
