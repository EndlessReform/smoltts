import mlx.core as mx
import mlx.nn as nn
from typing import Any, List


class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0  # Keep tracking offset for interface compatibility

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Update cache with new keys/values and return full concatenated cache."""
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.offset += keys.shape[2]
        return self.keys, self.values


def make_prompt_cache(model: nn.Module, is_fast: bool = False) -> List[Any]:
    """Construct the model's cache for use during generation."""
    if hasattr(model, "make_cache"):
        return model.make_cache()

    if is_fast:
        return [KVCache() for _ in range(len(model.fast_layers))]
    else:
        return [KVCache() for _ in range(len(model.layers))]
