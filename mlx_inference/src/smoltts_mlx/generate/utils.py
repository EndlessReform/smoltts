from pydantic import BaseModel, Field
from typing import Optional


class GenerationSettings(BaseModel):
    default_temp: float = Field(default=0.7)
    default_fast_temp: Optional[float] = Field(default=0.7)
    min_p: Optional[float] = Field(default=None)
    max_new_tokens: int = Field(default=1024)
