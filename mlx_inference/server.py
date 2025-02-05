from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import Response
import huggingface_hub
import json
import mlx.core as mx
from pathlib import Path
from pydantic import BaseModel, Field
import numpy as np
import rustymimi
import time
from tokenizers import Tokenizer
from typing import Literal, Union

from mlx_inference.lm.rq_transformer import (
    RQTransformerModelArgs,
    RQTransformer,
    TokenConfig,
)
from mlx_inference.lm.generate import generate_blocking
from mlx_inference.lm.utils.prompt import PromptEncoder
from mlx_inference.io.wav import pcm_to_wav_bytes
import mlx_inference
import mlx_inference.settings


def get_mimi_path():
    """Get Mimi tokenizer weights from Hugging Face."""
    repo_id = "kyutai/moshiko-mlx-bf16"
    filename = "tokenizer-e351c8d8-checkpoint125.safetensors"
    return huggingface_hub.hf_hub_download(repo_id, filename)


class SpeechRequest(BaseModel):
    model: str = Field(default="tts-1-hd")
    input: str
    voice: Union[str, int] = Field(default="alloy")
    response_format: Literal["wav"] = Field(default="wav")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO stop hard coding path
    with open("./settings/default_settings.json") as f:
        settings_json = json.loads(f.read())
        settings = mlx_inference.settings.ServerSettings(**settings_json)

    checkpoint_dir = Path(settings.checkpoint_dir)
    app.state.settings = settings
    model_type = settings.model_type

    load_start_time = time.time()
    print("Loading model configuration and tokenizer...")
    config = RQTransformerModelArgs.from_json_file(str(checkpoint_dir / "config.json"))
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    token_config = TokenConfig.from_tokenizer(
        model=model_type, tokenizer=tokenizer, config=config
    )

    print("Loading DualAR model...")
    model = RQTransformer(config, token_config, model_type)
    model_path = str(checkpoint_dir / "model.safetensors")
    model.load_weights(model_path, strict=True)
    mx.eval(model.parameters())
    model.eval()

    print("Downloading and initializing Mimi tokenizer...")
    mimi_path = get_mimi_path()
    app.state.mimi_tokenizer = rustymimi.Tokenizer(mimi_path)
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.prompt_encoder = PromptEncoder.from_model(tokenizer, model)

    load_end_time = time.time()
    print(f"Loaded model and config in {load_end_time - load_start_time:.3f} seconds")
    print(f"Default device: {mx.default_device()}")

    yield
    print("shutting down")


app = FastAPI(lifespan=lifespan)


@app.post("/v1/audio/speech")
async def handle_speech(item: SpeechRequest):
    speaker_id = item.voice if isinstance(item.voice, int) else 0
    sysprompt = app.state.prompt_encoder.encode_text_turn(
        "system", f"<|speaker:{speaker_id}|>"
    )
    user_prompt = app.state.prompt_encoder.encode_text_turn("user", item.input)
    assistant_prefix = app.state.prompt_encoder.encode_text_turn("assistant")
    prompt = mx.concat([sysprompt, user_prompt, assistant_prefix], axis=1)[
        mx.newaxis, :, :
    ]

    # Generate semantic tokens
    gen = generate_blocking(
        app.state.model, prompt, app.state.settings.generation, audio_only=True
    )

    # Convert to numpy and prepare for Mimi decoding
    tokens = np.array(gen).astype(np.uint32)

    # Decode to PCM using rustymimi
    start_time = time.time()
    pcm_data = app.state.mimi_tokenizer.decode(tokens)
    end_time = time.time()
    print(f"Took {end_time - start_time:.2f}s to decode")

    data = pcm_to_wav_bytes(pcm_data)

    return Response(
        data,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )
