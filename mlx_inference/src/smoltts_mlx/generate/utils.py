from pydantic import BaseModel
from typing import Optional


class GenerationSettings(BaseModel):
    default_temp: float = 0.7
    default_fast_temp: Optional[float] = 0.7
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: int = 1024
