import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, Field
from tokenizers import Tokenizer
from typing import Any, Optional, List, Literal, Tuple

from mlx_inference.lm.config import ModelType


class DualARModelArgs(BaseModel):
    model_type: str

    # Base transformer trunk
    vocab_size: int
    n_layer: int
    n_head: int
    n_local_heads: int
    head_dim: int = 64
    dim: int
    intermediate_size: int
    rope_base: float = 10_000
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False

    # Fast layers
    codebook_size: int = 2048
    num_codebooks: int = 8
    n_fast_layer: int = 4
    fast_dim: int
    fast_n_head: int
    fast_n_local_heads: int
    fast_head_dim: int
    fast_intermediate_size: int
    fast_attention_qkv_bias: bool = False
    fast_wte_embedding: Literal["flattened", "full"] = Field(default="flattened")
    """
    Flattened: Fish Speech default (indices are shared across RVQ levels), "full": RVQ is treated like main codebook
    """
    fast_positional_embedding: Literal["rope", "absolute"] = Field(default="rope")

    # meta
    use_gradient_checkpointing: bool = False

    @classmethod
    def from_json_file(cls, file_path: str) -> "DualARModelArgs":
        with open(file_path, "r") as f:
            return cls.model_validate_json(f.read())


class TokenConfig(BaseModel):
    im_end_id: int
    pad_id: int
    semantic_start_id: int
    semantic_end_id: Optional[int]

    @classmethod
    def from_tokenizer(
        cls, model: ModelType, tokenizer: Tokenizer, config: DualARModelArgs
    ):
        im_end = tokenizer.token_to_id("<|im_end|>")
        if im_end is None:
            raise ValueError("Tokenizer does not have <|im_end|>")

        if model.family == "dual_ar" or (
            model.family == "fish" and model.version == "1.5"
        ):
            semantic_start_id = tokenizer.token_to_id("<|semantic:0|>")
        else:
            semantic_start_id = tokenizer.token_to_id("<|semantic|>") or 5

        semantic_end_id = None
        pad_id = tokenizer.token_to_id("<|semantic|>") or 5

        if model.family == "dual_ar" or (
            model.family == "fish" and model.version == "1.5"
        ):
            semantic_end_id = tokenizer.token_to_id(
                f"<|semantic:{config.codebook_size - 1}|>"
            )

        return cls(
            **{
                "im_end_id": im_end,
                "semantic_start_id": semantic_start_id,
                "semantic_end_id": semantic_end_id,
                "pad_id": pad_id,
            }
        )


class DualARTransformer(nn.Module):
    def __init__(
        self, config: DualARModelArgs, token_config: TokenConfig, model_type: ModelType
    ):
        self.config = config
        self.token_config = token_config
        self.model_type = model_type

        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks, config.dim
        )
        self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        if not self.config.tie_word_embeddings:
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        else:
            self.output = None

        if config.fast_dim != config.dim:
            self.fast_project_in = nn.Linear(config.dim, config.fast_dim)
        else:
            self.fast_project_in = nn.Identity()

        if config.fast_wte_embedding == "full":
            self.fast_embeddings = nn.Embedding(
                config.codebook_size * (config.num_codebooks - 1), config.fast_dim
            )
        else:
            self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)

        if config.fast_positional_embedding == "absolute":
            self.fast_wpe = nn.Embedding(config.num_codebooks, config.fast_dim)
        else:
            self.fast_wpe = None

        self.fast_layers = [
            TransformerBlock(config, is_fast=True) for _ in range(config.n_fast_layer)
        ]
        self.fast_norm = nn.RMSNorm(config.fast_dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)

    def embed(self, x: mx.array) -> mx.array:
        semantic_tokens = x[:, 0, :]
        semantic_embeds = self.embeddings(semantic_tokens)[:, mx.newaxis, :]

        codebook_tokens = (
            x[:, 1:, :]
            + mx.arange(
                0,
                self.config.num_codebooks * self.config.codebook_size,
                self.config.codebook_size,
            )[:, mx.newaxis]
        )
        codebook_embeds = self.codebook_embeddings(codebook_tokens)

        if self.token_config.semantic_end_id is not None:
            emb_mask = (semantic_tokens >= self.token_config.semantic_start_id) & (
                semantic_tokens <= self.token_config.semantic_end_id
            )
        else:
            emb_mask = semantic_tokens == self.token_config.semantic_start_id

        codebook_embeds = codebook_embeds * emb_mask[:, :, mx.newaxis]
        return mx.concat([semantic_embeds, codebook_embeds], axis=1).sum(axis=1)

    # def __call__(self, ):
    def forward_generate(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> Tuple[mx.array, mx.array]:
        x = self.embed(inputs)
        mask = create_attention_mask(x, cache) if x.shape[1] > 1 else None

        for layer, layer_cache in zip(self.layers, cache or [None] * len(self.layers)):
            x = layer(x, mask=mask, cache=layer_cache)

        x = x[:, -1, :]  # Only take the last token for generation
        slow_out = self.norm(x)
        if self.output is not None:
            token_logits = self.output(slow_out)
        else:
            token_logits = self.embeddings.as_linear(slow_out)

        x = self.fast_project_in(x)
        return (token_logits, x)

    def forward_generate_fast(
        self,
        x: mx.array,
        cache: Optional[List[Any]] = None,
        input_pos: Optional[int] = 0,
    ) -> mx.array:
        """
        Assumes (bsz, seqlen=1, fast_dim)
        """
        mask = create_attention_mask(x, cache) if x.shape[1] > 1 else None
        if self.fast_wpe is not None:
            pos_emb = self.fast_wpe(mx.array([input_pos], dtype=mx.uint32))
            x += pos_emb

        for layer, layer_cache in zip(
            self.fast_layers, cache or [None] * len(self.fast_layers)
        ):
            x = layer(x, mask=mask, cache=layer_cache)

        fast_out = self.fast_norm(x)
        return self.fast_output(fast_out)


class TransformerBlock(nn.Module):
    def __init__(self, config: DualARModelArgs, is_fast: bool = False):
        super().__init__()
        self.attention = Attention(config, is_fast)
        self.feed_forward = MLP(config)
        self.ffn_norm = nn.RMSNorm(dims=config.dim, eps=config.norm_eps)
        self.attention_norm = nn.RMSNorm(config.dim, config.norm_eps)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        h = x + self.attention(self.attention_norm(x), mask=mask, cache=cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: DualARModelArgs, is_fast: bool = False):
        super().__init__()
        # GQA: groups split hidden dim evenly between them
        assert config.dim % config.n_head == 0

        if config.fast_positional_embedding == "rope" or not is_fast:
            self.rope = nn.RoPE(
                int(config.dim / config.n_head), traditional=True, base=config.rope_base
            )
        else:
            self.rope = None

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(
            input_dims=config.dim,
            output_dims=total_head_dim,
            bias=config.attention_qkv_bias,
        )
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        self.n_local_heads, self.n_head, self.head_dim, self.dim = (
            config.n_local_heads,
            config.n_head,
            config.head_dim,
            config.dim,
        )
        # Manually apply $\sqrt{d_k}$
        self.scale = config.head_dim**-0.5

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        bsz, seqlen, _ = x.shape
        qkv = self.wqkv(x)

        # Split qkv back to constituent sections
        kv_size = self.n_local_heads * self.head_dim
        raw = qkv.split([self.dim, self.dim + kv_size], axis=-1)
        q, k, v = raw
        q = q.reshape((bsz, seqlen, self.n_head, self.head_dim))
        k = k.reshape((bsz, seqlen, self.n_local_heads, self.head_dim))
        v = v.reshape((bsz, seqlen, self.n_local_heads, self.head_dim))

        q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))
        if cache is not None:
            if self.rope is not None:
                q = self.rope(q, offset=cache.offset)
                k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)

        else:
            q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))
            if self.rope is not None:
                q = self.rope(q)
                k = self.rope(k)

        output = mx.fast.scaled_dot_product_attention(
            q=q, k=k, v=v, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        return self.wo(output)


class MLP(nn.Module):
    def __init__(self, config: DualARModelArgs) -> None:
        super().__init__()

        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


def create_attention_mask(h: mx.array, cache: Optional[Any] = None):
    T = h.shape[1]
    if T > 1:
        offset = cache[0].offset if cache is not None and cache[0] is not None else 0
        mask = _create_causal_mask(T, offset=offset)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask


def _create_causal_mask(N: int, offset: int = 0):
    """Creates a causal mask for attention."""
    linds, rinds = mx.arange(offset + N), mx.arange(offset, offset + N)
    mask = (linds[:, None] < rinds[None]) * -1e9
    return mask
