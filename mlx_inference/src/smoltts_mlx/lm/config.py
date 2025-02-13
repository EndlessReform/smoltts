from pydantic import BaseModel
from typing import Literal, Optional


class ModelType(BaseModel):
    family: Literal["fish", "dual_ar"]
    version: Optional[Literal["1.5", "1.4", "1.2"]]
    codec: Literal["mimi", "1.4", "1.2"]
