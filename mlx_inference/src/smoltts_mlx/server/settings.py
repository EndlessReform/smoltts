# in mlx_inference/settings.py
from pathlib import Path
import os
import json
from pydantic import BaseModel, Field
from typing import Optional
from smoltts_mlx.lm.config import ModelType


class GenerationSettings(BaseModel):
    default_temp: float = Field(default=0.7)
    default_fast_temp: Optional[float] = Field(default=0.7)
    min_p: Optional[float] = Field(default=None)
    max_new_tokens: int = Field(default=1024)


class ServerSettings(BaseModel):
    checkpoint_dir: str
    generation: GenerationSettings
    model_type: ModelType

    @classmethod
    def get_settings(cls, config_path: Optional[str] = None) -> "ServerSettings":
        """Get settings from config file or create default in cache dir."""
        default_settings = {
            "checkpoint_dir": "./checkpoints",
            "model_type": {"family": "dual_ar", "codec": "mimi", "version": None},
            "generation": {
                "default_temp": 0.0,
                "default_fast_temp": 0.5,
                "min_p": 0.10,
                "max_new_tokens": 1024,
            },
        }

        if config_path:
            with open(config_path) as f:
                return cls(**json.loads(f.read()))

        # Use macOS cache dir
        config_dir = Path(os.path.expanduser("~/Library/Caches/smolltts/settings"))
        config_path = config_dir / "config.json"

        config_dir.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(default_settings, f, indent=2)
            return cls(**default_settings)

        with open(config_path) as f:
            return cls(**json.loads(f.read()))
