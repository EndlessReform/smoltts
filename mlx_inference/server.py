from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import Response
import huggingface_hub
import mlx.core as mx
from pathlib import Path
from pydantic import BaseModel, Field
import numpy as np
import rustymimi
import time
from tokenizers import Tokenizer
from typing import Literal, Union

from mlx_inference.model.dual_ar import DualARModelArgs, DualARTransformer, TokenConfig
from mlx_inference.model.config import ModelType
from mlx_inference.model.generate import generate_blocking
from mlx_inference.model.utils.prompt import PromptEncoder
from mlx_inference.io.wav import pcm_to_wav_bytes


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
    checkpoint_dir = Path("../inits/smoltts_byte_reference")
    model_type = ModelType(family="dual_ar", version=None, codec="mimi")

    load_start_time = time.time()
    print("Loading model configuration and tokenizer...")
    config = DualARModelArgs.from_json_file(str(checkpoint_dir / "config.json"))
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    token_config = TokenConfig.from_tokenizer(
        model=model_type, tokenizer=tokenizer, config=config
    )

    print("Loading DualAR model...")
    model = DualARTransformer(config, token_config, model_type)
    model_path = str(checkpoint_dir / "model.safetensors")
    model.load_weights(model_path, strict=True)
    model = model.apply(lambda p: p.astype(mx.float32))
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
    gen = generate_blocking(app.state.model, prompt, audio_only=True)

    # Convert to numpy and prepare for Mimi decoding
    tokens = np.array(gen).astype(np.uint32)

    # Decode to PCM using rustymimi
    start_time = time.time()
    pcm_data = app.state.mimi_tokenizer.decode(tokens)
    end_time = time.time()
    print(f"Took {end_time - start_time:.2f}s to decode")

    start_time = time.time()
    data = pcm_to_wav_bytes(pcm_data)
    end_time = time.time()
    print(f"Took {end_time - start_time:.5f}s to add headers")

    return Response(
        data,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )
