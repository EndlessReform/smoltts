from huggingface_hub import snapshot_download
import json
import os
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import Optional

from smoltts_mlx.lm.config import ModelType


class GenerationSettings(BaseModel):
    default_temp: float = Field(default=0.7)
    default_fast_temp: Optional[float] = Field(default=0.7)
    min_p: Optional[float] = Field(default=None)
    max_new_tokens: int = Field(default=1024)


class ServerSettings(BaseModel):
    model_id: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    generation: GenerationSettings
    model_type: ModelType

    @model_validator(mode="after")
    def validate_model_source(self):
        if self.model_id is not None and self.checkpoint_dir is not None:
            raise ValueError("Cannot specify both model_id and checkpoint_dir")
        if self.model_id is None and self.checkpoint_dir is None:
            raise ValueError("Must specify either model_id or checkpoint_dir")

        return self

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

    def get_checkpoint_dir(self) -> Path:
        if self.checkpoint_dir is not None:
            return Path(self.checkpoint_dir)
        else:
            # guaranteed to exist by validator above
            hf_repo_path = snapshot_download(self.model_id)
            return Path(hf_repo_path)
