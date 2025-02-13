import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, Field
from typing import Any, List, Optional


from smoltts_mlx.lm.rq_transformer import create_attention_mask


class MimiTransformerConfig(BaseModel):
    # NOTE: Leaving out norm, RoPE call, ffn, attn bias because this is pointless, it's standard LLaMA
    d_model: int = Field(default=512)
    num_heads: int = Field(default=8)
    """
    Mimi v1 is NOT GQA so I'm deliberately setting this one only
    """
    head_dim: int = Field(default=64)
    num_layers: int = Field(default=8)
    causal: bool = Field(default=True)
    norm_first: bool = Field(default=True)
    layer_scale: Optional[float] = Field(default=0.01)
    context: int = Field(default=250)
    conv_kernel_size: int = Field(default=5)
    use_conv_bias: bool = Field(default=True)
    use_conv_block: bool = Field(default=False)
    max_period: int = Field(default=10_000)

    dim_feedforward: int = Field(default=2048)
    kv_repeat: int = Field(default=1)
    max_seq_len: int = Field(default=8192)
    rope_theta: float = Field(default=10_000)


class MimiMLP(nn.Module):
    def __init__(self, config: MimiTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.dim_feedforward, bias=False)
        self.fc2 = nn.Linear(config.dim_feedforward, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class MimiAttention(nn.Module):
    def __init__(self, config: MimiTransformerConfig):
        self.q_proj = nn.Linear(
            config.d_model, config.num_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.d_model, config.num_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.d_model, config.num_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_heads * config.head_dim, config.d_model, bias=False
        )
        self.rope = nn.RoPE(
            int(config.d_model / config.num_heads),
            traditional=False,
            base=config.rope_theta,
        )

        self.scaling = config.head_dim**-0.5
        self.n_head = config.num_heads
        self.head_dim = config.head_dim

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape((bsz, seqlen, self.n_head, self.head_dim))
        k = k.reshape((bsz, seqlen, self.n_head, self.head_dim))
        v = v.reshape((bsz, seqlen, self.n_head, self.head_dim))

        q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))
        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        output = mx.fast.scaled_dot_product_attention(
            q=q, k=k, v=v, scale=self.scaling, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        return self.o_proj(output)


class MimiLayerScale(nn.Module):
    def __init__(self, config: MimiTransformerConfig):
        super().__init__()
        self.scale = mx.full((config.d_model,), config.layer_scale)

    def __call__(self, x: mx.array) -> mx.array:
        # TODO check this
        return x * self.scale


class MimiTransformerLayer(nn.Module):
    def __init__(self, config: MimiTransformerConfig):
        self.mlp = MimiMLP(config)
        # Leaving out eps since the MLX default is identical and the weights won't change
        self.input_layernorm = nn.LayerNorm(config.d_model)
        self.post_attention_layernorm = nn.LayerNorm(config.d_model)
        self.self_attn = MimiAttention(config)

        self.self_attn_layer_scale = MimiLayerScale(config)
        self.mlp_layer_scale = MimiLayerScale(config)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        residual = x
        hidden_states = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = residual + self.self_attn_layer_scale(hidden_states)

        residual = h
        hidden_states = self.post_attention_layernorm(h)
        hidden_states = self.mlp(hidden_states)
        h = residual + self.mlp_layer_scale(hidden_states)
        return h


class MimiTransformer(nn.Module):
    def __init__(self, config: MimiTransformerConfig):
        super().__init__()
        self.layers = [MimiTransformerLayer(config) for _ in range(config.num_layers)]
        self.config = config

    def __call__(
        self,
        x: mx.array,
        cache: Optional[List[Any]] = None,
    ):
        mask = create_attention_mask(x, cache) if x.shape[1] > 1 else None

        for layer, layer_cache in zip(self.layers, cache or [None] * len(self.layers)):
            x = layer(x, mask=mask, cache=layer_cache)

        return x
