import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple


class SeanetConfig(BaseModel):
    dimension: int = Field(default=512)
    channels: int = Field(default=1)
    n_filters: int = Field(default=64)
    n_residual_layers: int = Field(default=1)
    activation: Literal["elu"] = Field(default="elu")
    compress: int = Field(default=2)
    dilation_base: int = Field(default=2)
    disable_norm_outer_blocks: bool = Field(default=False)
    final_activation: Optional[str] = Field(default=None)
    kernel_size: int = Field(default=7)
    residual_kernel_size: int = Field(default=3)
    last_kernel_size: int = Field(default=3)
    lstm: bool = Field(default=False)
    norm: Literal["weight_norm"] = Field(default="weight_norm")
    pad_mode: Literal["constant"] = Field(default="constant")
    ratios: List[int] = Field(default=[8, 6, 5, 4])
    true_skip: bool = Field(default=True)


class SeaNetResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        k_sizes_and_dilations: List[Tuple[int, int]],
        activation: str,
        norm: Optional[str],
        causal: bool,
        pad_mode: str,
        compress: int,
        true_skip: bool,
    ):
        super().__init__()
        hidden = dim / compress
        for i, (k_size, dilation) in enumerate(k_sizes_and_dilations):
            # Projecting up in the block, then down, similar to transformer MLP
            in_c = dim if i == 0 else hidden
            out_c = dim if i == len(k_sizes_and_dilations) - 1 else hidden

        pass
