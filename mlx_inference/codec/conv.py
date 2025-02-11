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


@mx.compile
def causal_pad1d(
    x: mx.array, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0
) -> mx.array:
    if x.ndim < 2:
        raise ValueError(
            "Input tensor must have at least 2 dimensions (seq_len, channels)."
        )

    padding_left, padding_right = paddings

    # Create a padding tuple that pads only the second-to-last dimension (seqlen)
    pad_tuple = [(0, 0)] * x.ndim
    pad_tuple[-2] = (padding_left + padding_right, 0)
    pad_tuple = tuple(pad_tuple)

    if mode != "reflect":
        out = mx.pad(x, pad_tuple, mode, value)
        return out

    # Handle reflect mode with possible extra padding
    length = x.shape[-2]
    max_pad = max(padding_left, padding_right)
    extra_pad = 0

    if length <= max_pad:
        extra_pad = max_pad - length + 1
        # Apply extra padding to the seqlen dimension
        x_pad = [(0, 0)] * x.ndim
        x_pad[-2] = (0, extra_pad)
        x = mx.pad(x, tuple(x_pad), "constant")

    padded = mx.pad(x, pad_tuple, mode, value)

    if extra_pad > 0:
        # Slice to remove the extra padding added for reflection
        slices = [slice(None)] * x.ndim
        slices[-2] = slice(None, padded.shape[-2] - extra_pad)
        padded = padded[tuple(slices)]

    return padded


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

        self._stream_prev_in: Optional[mx.array] = None
        self._left_pad_applied = False

    def reset_state(self):
        """
        Clears any leftover input from previous streaming steps.
        Call this before processing a brand new stream.
        """
        self._stream_prev_in = None
        self._left_pad_applied = False

    def _get_extra_padding_for_conv1d(self, x: mx.array) -> int:
        length = x.shape[-2]  # Use the seqlen dimension
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = math.ceil(n_frames) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total

        return ideal_length - length

    def __call__(self, x: mx.array) -> mx.array:
        extra_padding = self._get_extra_padding_for_conv1d(x)
        x = causal_pad1d(
            x,
            (self.padding_left, self.padding_right + extra_padding),
            self.pad_mode,
        )
        x = self.conv(x)
        return x

    def step(self, x: mx.array) -> Optional[mx.array]:
        """
        Streaming forward pass: processes chunk x and merges with leftover output
        from the previous step to avoid pops.
        """
        effective_k_size = (self.kernel_size - 1) * self.dilation + 1
        if not self._left_pad_applied:
            self._left_pad_applied = True
            padding_total = effective_k_size - self.stride
            x = mx.pad(x, [(0, 0), (padding_total, 0), (0, 0)])

        # 1) Restore previous input
        if self._stream_prev_in is not None:
            x_long = mx.concat([self._stream_prev_in, x], axis=-2)
        else:
            x_long = x

        num_frames = int(
            max(x_long.shape[-2] + self.stride - effective_k_size, 0) / self.stride
        )
        if num_frames == 0:
            # print(
            #     f"Stride: {self.stride}, k_size: {effective_k_size}, length: {x_long.shape[-2]}"
            # )
            return None
            # raise ValueError("Encode not implemented")

        offset = num_frames * self.stride
        self._stream_prev_in = x_long[:, offset:, :]
        # print(
        #     f"Prev in offset: {offset}, length: {self._stream_prev_in.shape}, new_length: {new_length}"
        # )

        in_length = (num_frames - 1) * self.stride + effective_k_size
        # print(
        #     f"in_l: {in_length}, k_size: {effective_k_size}, num_frames: {num_frames}, stride: {self.stride}"
        # )
        xs = x_long[:, :in_length, :]
        return self.conv(xs)


class MimiConvTranspose1d(nn.Module):
    def __init__(
        self,
        config: SeanetConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
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

        self.stride = stride
        self.kernel_size = kernel_size
        padding_total = kernel_size - stride
        self.padding_right = math.ceil(padding_total * self.trim_right_ratio)
        self.padding_left = padding_total - self.padding_right

        self._stream_prev_out: Optional[mx.array] = None

    def reset_state(self):
        """
        Clears leftover output from previous streaming steps.
        """
        self._stream_prev_out = None

    def __call__(self, x: mx.array):
        x = self.conv(x)
        end = x.shape[-2] - self.padding_right
        x = x[:, self.padding_left : end, :]
        return x

    def step(self, x: mx.array) -> mx.array:
        ys = self.conv(x)
        ot = ys.shape[-2]
        if self._stream_prev_out is not None:
            prev_len = self._stream_prev_out.shape[-2]
            # Remove the bias to avoid it happening multiple times
            if self.conv.bias is not None:
                # TODO this is almost certainly wrong and needs the reshape.
                # Fuck you future self, kill yourself, die bitch, I am sick of this fucking bullshit
                self._stream_prev_out -= self.conv.bias
            ys1 = ys[:, :prev_len, :] + self._stream_prev_out
            ys2 = ys[:, prev_len:, :]
            ys = mx.concat([ys1, ys2], axis=-2)
        invalid_steps = self.kernel_size - self.stride
        split_point = ot - invalid_steps
        ys, prev_ys = ys[:, :split_point, :], ys[:, split_point:, :]
        self._stream_prev_out = prev_ys
        return ys

    # def step(self, x: mx.array) -> mx.array:
    #     """
    #     Streaming forward pass: processes chunk x and merges with leftover output
    #     from the previous step to avoid pops.
    #     """
    #     # 1) Perform the transposed conv
    #     y_full = self.conv(x)  # (batch, out_time, out_channels)

    #     # 2) Merge leftover from previous chunk (if any)
    #     if self._stream_prev_out is not None:
    #         overlap_sz = self._stream_prev_out.shape[-2]
    #         # Combine leftover with the first overlap_sz portion of y_full
    #         # e.g. y_merged_start = y_full[..., :overlap_sz, :] + self._stream_prev_out
    #         # then cat with the remainder
    #         if overlap_sz <= y_full.shape[-2]:
    #             y_merged_start = y_full[..., :overlap_sz, :] + self._stream_prev_out
    #             y_merged_tail = y_full[..., overlap_sz:, :]
    #             y_full = mx.concat([y_merged_start, y_merged_tail], axis=-2)
    #         else:
    #             # If leftover is larger than new output, just add in place
    #             y_full = self._stream_prev_out[..., : y_full.shape[-2], :] + y_full
    #     merged_len = y_full.shape[-2]

    #     # 3) Trim the same as __call__ does
    #     end = merged_len - self.padding_right
    #     y_full = y_full[:, self.padding_left : end, :]

    #     # 4) Figure out how many frames we want to keep for the next chunk
    #     #    Typically leftover_out = (kernel_size - stride)
    #     #    That portion is the "invalid tail" that should be overlapped next time
    #     leftover_sz = max(0, self.kernel_size - self.stride)
    #     if leftover_sz > 0 and y_full.shape[-2] > leftover_sz:
    #         # the final leftover_sz frames are for next overlap
    #         new_end = y_full.shape[-2] - leftover_sz
    #         out_final = y_full[:, :new_end, :]
    #         self._stream_prev_out = y_full[:, new_end:, :]
    #     else:
    #         # no leftover or not enough frames to separate
    #         out_final = y_full
    #         self._stream_prev_out = None

    #     return out_final


class MeaninglessConvPassthrough(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        """
        This is necessary to load in the weights without using the MLX nn wrapper
        """
        super().__init__()
        self.weight = mx.zeros([in_channels, kernel_size, out_channels])


class GroupedConvTranspose1d(nn.Module):
    def __init__(
        self,
        config: SeanetConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias=False,
    ):
        super().__init__()
        self.trim_right_ratio = config.trim_right_ratio
        channels_per_group = out_channels
        self.conv = MeaninglessConvPassthrough(
            in_channels,
            out_channels
            // channels_per_group,  # This becomes 1 when groups == in_channels
            kernel_size,
        )
        padding_total = kernel_size - stride
        # Due to the dilation
        self.padding_right = math.ceil(padding_total * self.trim_right_ratio)
        self.padding_left = padding_total - self.padding_right
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self._conv_weight = None

    @property
    def conv_weight(self):
        if self._conv_weight is None:
            # Torch collapses the groups on the OUTPUT, but MLX collapses on the INPUT
            self._conv_weight = self.conv.weight.reshape(
                self.groups, self.kernel_size, self.in_channels // self.groups
            )
        return self._conv_weight

    def __call__(self, x: mx.array):
        x = mx.conv_transpose1d(
            x,
            self.conv_weight,
            padding=0,
            stride=self.stride,
            groups=self.groups,
        )

        # print(f"IMMEDIATELY AFTER upsample conv: {x.shape}")
        end = x.shape[-2] - self.padding_right
        x = x[:, self.padding_left : end, :]
        return x
