import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional

from smoltts_mlx.codec.conv import SeanetConfig, MimiConv1d, MimiConvTranspose1d


class MimiResnetBlock(nn.Module):
    def __init__(self, config: SeanetConfig, dim: int, dilations: List[int]):
        super().__init__()
        assert len(dilations) == 2, (
            "Number of kernel sizes should match number of dilations"
        )
        kernel_sizes = (config.residual_kernel_size, 1)
        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU()]
            block += [
                MimiConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)
            ]

        self.block = block

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        for layer in self.block:
            x = layer(x)
        return residual + x

    def step(self, x: mx.array) -> Optional[mx.array]:
        residual = x
        for layer in self.block:
            if callable(getattr(layer, "step", None)):
                step = layer.step(x)
                if step is not None:
                    x = step
                else:
                    return None
            else:
                x = layer(x)
        return residual + x

    def reset_state(self):
        for layer in self.block:
            if callable(getattr(layer, "reset_state", None)):
                layer.reset_state()


class MimiEncoder(nn.Module):
    """
    SEANet encoder as used by Mimi.
    """

    def __init__(self, config: SeanetConfig):
        super().__init__()
        model = [
            MimiConv1d(config, config.channels, config.n_filters, config.kernel_size)
        ]
        scaling = 1

        for ratio in reversed(config.ratios):
            current_scale = scaling * config.n_filters
            for j in range(config.n_residual_layers):
                model += [
                    MimiResnetBlock(config, current_scale, [config.dilation_base**j, 1])
                ]
            model += [nn.ELU()]
            model += [
                MimiConv1d(
                    config,
                    current_scale,
                    current_scale * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            ]
            scaling *= 2

        model += [nn.ELU()]
        model += [
            MimiConv1d(
                config,
                scaling * config.n_filters,
                config.dimension,
                config.last_kernel_size,
            )
        ]
        self.layers = model

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class MimiDecoder(nn.Module):
    """
    SEANet decoder as used by Mimi
    """

    def __init__(self, config: SeanetConfig):
        super().__init__()
        scaling = int(2 ** len(config.ratios))
        model = [
            MimiConv1d(
                config, config.dimension, scaling * config.n_filters, config.kernel_size
            )
        ]

        for ratio in config.ratios:
            current_scale = scaling * config.n_filters
            model += [nn.ELU()]
            model += [
                MimiConvTranspose1d(
                    config,
                    current_scale,
                    current_scale // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            ]
            for j in range(config.n_residual_layers):
                model += [
                    MimiResnetBlock(
                        config, current_scale // 2, [config.dilation_base**j, 1]
                    )
                ]
            scaling //= 2

        model += [nn.ELU()]
        model += [
            MimiConv1d(
                config, config.n_filters, config.channels, config.last_kernel_size
            )
        ]
        self.layers = model

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def step(self, x: mx.array) -> Optional[mx.array]:
        for i, layer in enumerate(self.layers):
            if callable(getattr(layer, "step", None)):
                step = layer.step(x)
                if step is not None:
                    x = step
                else:
                    return None
            else:
                x = layer(x)
        return x

    def reset(self):
        for layer in self.layers:
            if callable(getattr(layer, "reset_state", "None")):
                layer.reset_state()
