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


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_k_sampling(
    logprobs: mx.array,
    top_k: int,
    temperature: float = 1.0,
) -> mx.array:
    if top_k < 1:
        raise ValueError("`top_k` must be at least 1")

    # Adjust logits by temperature.
    logprobs = logprobs * (1 / temperature)

    # Sort token indices by descending log probability.
    sorted_indices = mx.argsort(-logprobs).squeeze(0)
    sorted_logprobs = logprobs[..., sorted_indices]

    # Mask tokens outside the top-k by setting their logprobs to -inf.
    tokens_to_remove = mx.zeros_like(sorted_logprobs) == 1
    tokens_to_remove[..., top_k:] = True
    filtered_logprobs = mx.where(tokens_to_remove, -float("inf"), sorted_logprobs)

    # Sample a token from the filtered pool and map it back to the original index.
    sampled_token = mx.random.categorical(filtered_logprobs)
    return sorted_indices[sampled_token]
