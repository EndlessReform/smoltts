import math
import mlx.core as mx
from functools import partial


# From MLX examples
@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def min_p_sampling(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
    temperature=1.0,
) -> mx.array:
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )

    logprobs = logprobs * (1 / temperature)
    # Sort indices in decreasing order
    sorted_indices = mx.argsort(-logprobs).squeeze(0)
    sorted_logprobs = logprobs[..., sorted_indices]
    # Get top probability
    top_logprobs = logprobs[..., sorted_indices]
    # Calculate min-p threshold
    scaled_min_p = top_logprobs + math.log(min_p)
    # Mask tokens below threshold
    tokens_to_remove = sorted_logprobs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False
    # Create filtered token pool
    selected_logprobs = mx.where(tokens_to_remove, -float("inf"), sorted_logprobs)
    # Sample and return token
    sorted_token = mx.random.categorical(selected_logprobs)
    return sorted_indices[sorted_token]
