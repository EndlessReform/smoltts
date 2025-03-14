from pydantic import BaseModel
from typing import Literal, Optional


class ModelType(BaseModel):
    family: Literal["fish", "dual_ar", "csm"]
    version: Optional[Literal["1.5", "1.4", "1.2"]]
    codec: Literal["mimi", "1.4", "1.2"]

    @classmethod
    def smoltts_v0(cls):
        return cls(family="dual_ar", version=None, codec="mimi")

    @classmethod
    def csm_1b(cls):
        return cls(family="csm", version=None, codec="mimi")
