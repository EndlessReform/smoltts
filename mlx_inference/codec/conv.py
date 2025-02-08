import math
import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple


class SeanetConfig(BaseModel):
    dimension: int = Field(default=512)
    channels: int = 1
    n_filters: int = 64
    n_residual_layers: int = 1
    compress: int = 2
    dilation_base: int = 2
    disable_norm_outer_blocks: int = 0
    kernel_size: int = 7
    residual_kernel_size: int = 3
    last_kernel_size: int = 3
    # lstm: 0,
    # pad_mode: conv::PadMode::Constant,
    ratios: List[int] = [8, 6, 5, 4]
    trim_right_ratio: float = 1.0
    sampling_rate: float = 24_000.0
    upsample_groups: int = 512


def causal_pad1d(
    x: mx.array, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0
) -> mx.array:
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if not mode == "reflect":
        return mx.pad(x, paddings, mode, value)

    max_pad = max(padding_left, padding_right)
    extra_pad = 0
    if length <= max_pad:
        extra_pad = max_pad - length + 1
        x = mx.pad(x, (0, extra_pad))
    padded = mx.pad(x, paddings, mode, value)
    end = padded.shape[-1] - extra_pad
    return padded[:, :end]


class MimiConv1d(nn.Module):
    def __init__(
        self,
        config: SeanetConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: Optional[str] = None,
    ):
        super().__init__()
        self.pad_mode = pad_mode if pad_mode is not None else "constant"
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.dilation = dilation
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        self.kernel_size = effective_kernel_size
        self.padding_total = effective_kernel_size - stride

        self.padding_right = self.padding_total // 2
        self.padding_left = self.padding_total - self.padding_right

    def _get_extra_padding_for_conv1d(self, x: mx.array) -> int:
        length = x.shape[-1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = math.ceil(n_frames) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total

        return ideal_length - length

    def __call__(self, x: mx.array) -> mx.array:
        extra_padding = self._get_extra_padding_for_conv1d(x)
        x = causal_pad1d(x, (self.padding_total, extra_padding), self.pad_mode)
        x = self.conv(x)
        return x


class MimiConvTranspose1d(nn.Module):
    def __init__(
        self,
        config: SeanetConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias=True,
    ):
        super().__init__()
        self.trim_right_ratio = config.trim_right_ratio
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias=bias,
        )

        padding_total = kernel_size - stride
        self.padding_right = math.ceil(padding_total * self.trim_right_ratio)
        self.padding_left = padding_total - self.padding_right

    def __call__(self, x: mx.array):
        x = self.conv(x)
        end = x.shape[-1] - self.padding_right
        x = x[:, self.padding_left : end]
        return x


class GroupedConvTranspose1d(nn.Module):
    def __init__(
        self,
        config: SeanetConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias=True,
    ):
        super().__init__()
        self.trim_right_ratio = config.trim_right_ratio
        self.conv_weight = mx.zeros([in_channels, out_channels, kernel_size])

        padding_total = kernel_size - stride
        self.padding_right = math.ceil(padding_total * self.trim_right_ratio)
        self.padding_left = padding_total - self.padding_right
        self.groups = groups

    def __call__(self, x: mx.array):
        x = mx.conv_general(
            x,
            self.conv_weight,
            padding=(self.padding_left, self.padding_right),
            groups=self.groups,
            flip=False,
        )
        end = x.shape[-1] - self.padding_right
        x = x[:, self.padding_left : end]
        return x
