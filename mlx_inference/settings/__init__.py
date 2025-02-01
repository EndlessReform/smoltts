from pydantic import BaseModel, Field
from typing import Optional
from mlx_inference.lm.config import ModelType


class GenerationSettings(BaseModel):
    default_temp: float = Field(default=0.7)
    default_fast_temp: Optional[float] = Field(default=0.7)
    min_p: Optional[float] = Field(default=None)
    max_new_tokens: int = Field(default=1024)


class ServerSettings(BaseModel):
    checkpoint_dir: str
    generation: GenerationSettings
    model_type: ModelType
