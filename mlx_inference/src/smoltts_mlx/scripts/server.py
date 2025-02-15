import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
import time

from smoltts_mlx import SmolTTS
from smoltts_mlx.server.tts_core import TTSCore
from smoltts_mlx.server.routes import openai, elevenlabs
from smoltts_mlx.server.settings import ServerSettings


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    checkpoint_dir = settings.get_checkpoint_dir()

    load_start_time = time.time()
    print("Loading model...")
    model = SmolTTS(checkpoint_dir=checkpoint_dir)
    app.state.tts_core = TTSCore(
        model=model,
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
