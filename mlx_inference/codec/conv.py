import mlx.core as mx
import mlx.nn as nn


class NormConv1D(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        The upstream uses weight norm for pretraining, but it's merged in the .safetensors file,
        so this is just here to make the weight dict keys match
        """
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)

    def __call__(self, x):
        x = self.conv(x)
        return x


class StreamingConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        pad_mode: str = "reflect",
    ):
        """
        This is not streaming; merely keeping the name for easier porting with upstream
        """
        super().__init__()
        self.conv = NormConv1D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def __call__(self, x):
        pass
