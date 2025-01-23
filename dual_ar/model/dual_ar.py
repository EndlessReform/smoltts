import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class BaseModelArgs:
    model_type: str = "base"

    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = 16_384
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: int = 10000
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False

    # Codebook configs
    codebook_size: int = 160
    num_codebooks: int = 4

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # Initialize the model
    initializer_range: float = 0.02

    # Dummy vars
    is_reward_model: bool = False
    share_codebook_embeddings: bool = True
    scale_codebook_embeddings: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @staticmethod
    def from_pretrained(pathname: str):
        path = Path(pathname)

        if path.is_dir():
            path = path / "config.json"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cls = DualARModelArgs

        return cls(**data)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, ensure_ascii=False)


@dataclass
class DualARModelArgs(BaseModelArgs):
    model_type: str = "dual_ar"
    fast_dim: int = 1024
    n_fast_layer: int = 4
    fast_n_head: int = 16
    fast_n_local_heads: Optional[int] = None
    fast_head_dim: Optional[int] = None
    fast_intermediate_size: Optional[int] = None
    fast_attention_qkv_bias: Optional[bool] = None

    def __post_init__(self):
        super().__post_init__()

        self.fast_dim = self.fast_dim or self.dim
        self.fast_n_head = self.fast_n_head or self.n_head
        self.fast_n_local_heads = self.fast_n_local_heads or self.n_local_heads
        self.fast_head_dim = self.fast_head_dim or self.head_dim
        self.fast_intermediate_size = (
            self.fast_intermediate_size or self.intermediate_size
        )
        self.fast_attention_qkv_bias = (
            self.fast_attention_qkv_bias
            if self.fast_attention_qkv_bias is not None
            else self.attention_qkv_bias
        )


@dataclass
class TransformerForwardResult:
    token_logits: Tensor
    codebook_logits: Tensor


@dataclass
class BaseTransformerForwardResult:
    logits: Tensor
    hidden_states: Tensor


class BaseTransformer(nn.Module):
    def __init__(
        self,
        config: BaseModelArgs,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        # TODO: handle cases where we don't do this
        SEMANTIC_TOKENS = [f"<|semantic:{i}|>" for i in range(0, config.codebook_size)]
        self.semantic_token_ids = tokenizer.encode("".join(SEMANTIC_TOKENS))
        semantic_token_ids_tensor = torch.tensor(self.semantic_token_ids)
        if not (torch.diff(semantic_token_ids_tensor) == 1).all():
            raise ValueError("Semantic token IDs must be contiguous")

        self.register_buffer(
            "semantic_token_start", semantic_token_ids_tensor[0], persistent=False
        )
        self.register_buffer(
            "semantic_token_end", semantic_token_ids_tensor[-1], persistent=False
        )
        offset = torch.arange(
            0,
            self.config.num_codebooks * self.config.codebook_size,
            self.config.codebook_size,
        ).unsqueeze(1)
        self.register_buffer("semantic_offset", offset, persistent=False)

        # Slow transformer
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.dim,
        )
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks,
            config.dim,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=True) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if self.config.tie_word_embeddings is False:
            self.output = nn.Linear(
                config.dim,
                config.vocab_size,
                bias=False,
            )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.max_seq_len,
                config.dim // config.n_head,
                config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.max_seq_len,
                    config.max_seq_len,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )
        self.max_seq_len = config.max_seq_len if config.max_seq_len is not None else -1

        if init_weights:
            self.apply(self._init_weights)

    @torch.compile
    def embed(self, x: Tensor) -> Tensor:
        # Start with base token embedding
        text_embeds = self.embeddings(x[:, 0, :])

        # Get all codebook embeddings
        vq_embeds = self.codebook_embeddings(x[:, 1:, :] + self.semantic_offset)
        vq_embeds_sum = vq_embeds.sum(dim=1)

        # Mask summed embeddings where we don't have semantic tokens
        # vq_embeds_sum[~torch.isin(x[:, 0], self.semantic_token_tensor)] = 0
        vq_embeds_sum[
            ~(
                (x[:, 0] >= self.semantic_token_start)
                & (x[:, 0] <= self.semantic_token_end)
            )
        ] = 0

        return text_embeds + vq_embeds_sum

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> BaseTransformerForwardResult:
        seq_len = inp.size(2)

        # Here we want to merge the embeddings of the codebooks
        x = self.embed(inp)

        freqs_cis = self.freqs_cis[:seq_len]

        # Not that the causal mask here follows the definition of scaled_dot_product_attention
        # That is, FALSE means masked out
        # To maintain consistency, key_padding_mask use TRUE to mask out
        mask = None
        if key_padding_mask is not None:
            mask = self.causal_mask[None, None, :seq_len, :seq_len]  # (B, N, Q, K)
            mask = mask & key_padding_mask[:, None, None, :].logical_not()

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=True)
            else:
                x = layer(x, freqs_cis, mask)

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @staticmethod
    def from_pretrained(
        path: str,
        load_weights: bool = False,
        weight_override=None,
        max_length: Optional[int] = None,
        rope_base: Optional[int] = None,
    ) -> "BaseTransformer":
        config = BaseModelArgs.from_pretrained(str(path))
        if max_length is not None:
            config.max_seq_len = max_length
            print(f"Override max_seq_len to {max_length}")

        if rope_base is not None:
            config.rope_base = rope_base
            print(f"Override rope_base to {rope_base}")

        model_cls = DualARTransformer

        tokenizer = AutoTokenizer.from_pretrained(str(path))

        # TODO: logging
        print(f"Loading model from {path}, config: {config}")
        model = model_cls(config, tokenizer=tokenizer)

        if load_weights is False:
            print("Randomly initialized model")
        else:
            # TODO stop hard-coding this
            weights = torch.load(
                Path(path) / "model.pth",
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )

            # Verify the name and shape of parameters since strict=False in load_state_dict.
            for k, v in model.named_parameters():
                if k not in weights:
                    print(f"No weight for {k}")
                elif v.shape != weights[k].shape:
                    print(f"Shape mismatch for {k}: {v.shape} vs {weights[k].shape}")

            err = model.load_state_dict(weights, strict=False, assign=True)
            print(f"Loaded weights with error: {err}")

        return model

    def save_pretrained(self, pathname: str, drop_lora: bool = False):
        path = Path(pathname)
        path.mkdir(parents=True, exist_ok=True)

        self.config.save(str(path / "config.json"))
        state_dict = self.state_dict()

        torch.save(state_dict, path / "model.pth")
        self.tokenizer.save_pretrained(path)


class DualARTransformer(BaseTransformer):
    def __init__(self, config: DualARModelArgs, tokenizer: AutoTokenizer) -> None:
        super().__init__(config, init_weights=False, tokenizer=tokenizer)

        # Project to fast dim if needed
        if config.fast_dim is not None and config.fast_dim != config.dim:
            self.fast_project_in = nn.Linear(config.dim, config.fast_dim)
        else:
            self.fast_project_in = nn.Identity()

        # Fast transformer
        self.fast_embeddings = nn.Embedding(
            config.codebook_size * (config.num_codebooks - 1), config.fast_dim
        )
        offset = torch.arange(
            0,
            self.config.codebook_size * (self.config.num_codebooks - 1),
            self.config.codebook_size,
        ).unsqueeze(1)
        self.register_buffer("codebook_offset", offset, persistent=False)

        # The equivalent bs is so large that sdpa doesn't work
        override_config = dataclasses.replace(
            config,
            dim=config.fast_dim,
            n_head=config.fast_n_head,
            n_local_heads=config.fast_n_local_heads,
            head_dim=config.fast_head_dim,
            intermediate_size=config.fast_intermediate_size,
            attention_qkv_bias=config.fast_attention_qkv_bias,
        )

        self.fast_layers = nn.ModuleList(
            TransformerBlock(override_config, use_sdpa=False)
            for _ in range(config.n_fast_layer)
        )
        self.fast_norm = RMSNorm(config.fast_dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(
            config.fast_dim,
            config.codebook_size,
            bias=False,
        )

        self.register_buffer(
            "fast_freqs_cis",
            precompute_freqs_cis(
                config.num_codebooks,
                config.fast_dim // config.fast_n_head,
                config.rope_base,
            ),
            persistent=False,
        )
        self.apply(self._init_weights)

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        parent_result = super().forward(inp, key_padding_mask)
        token_logits = parent_result.logits
        x = parent_result.hidden_states
        x = self.fast_project_in(x)

        # Fast transformer
        fast_seq_len = self.config.num_codebooks
        fast_mask = self.causal_mask[
            None, None, :fast_seq_len, :fast_seq_len
        ]  # (B, N, Q, K)

        # Drop the last token and rotate left
        codebooks = inp[:, 1:-1, 1:]
        codebooks = codebooks + self.codebook_offset
        codebooks = F.pad(codebooks, (0, 1), value=0)
        codebook_embeddings = self.fast_embeddings(codebooks)
        x = torch.cat([x[:, None], codebook_embeddings], dim=1)
        b, s = x.size(0), x.size(2)
        x = rearrange(x, "b n s d -> (b s) n d")  # flatten the batch and seq_len

        # Remove padded part
        codebooks = rearrange(codebooks, "b n s -> (b s) n")
        codebook_mask = (codebooks == 0).all(dim=-1)

        if torch.all(codebook_mask):
            # If all codebooks are padded, we keep first 8 to make sure the model runs
            codebook_mask[:8] = False

        x_bs, x_len = x.size(0), x.size(1)
        x = x[~codebook_mask]

        for layer in self.fast_layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(
                    layer, x, self.fast_freqs_cis, fast_mask, use_reentrant=True
                )
            else:
                x = layer(x, self.fast_freqs_cis, fast_mask)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)
        codebook_logits = self.fast_output(fast_out)

        # Re-pad the codebook_logits
        buffer = torch.zeros(
            x_bs,
            x_len,
            codebook_logits.size(-1),
            device=codebook_logits.device,
            dtype=codebook_logits.dtype,
        )
        buffer[~codebook_mask] = codebook_logits
        codebook_logits = buffer

        assert codebook_logits.shape[1] == self.config.num_codebooks
        codebook_logits = rearrange(
            codebook_logits,
            "(b s) n d -> b s n d",
            b=b,
            s=s,
            n=self.config.num_codebooks,
        )

        return TransformerForwardResult(
            token_logits=token_logits,
            codebook_logits=codebook_logits,
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True) -> None:
        super().__init__()
        self.attention = Attention(config, use_sdpa=use_sdpa)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(
            config.dim, total_head_dim, bias=config.attention_qkv_bias
        )
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.use_sdpa = use_sdpa
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            # No third party attn_mask here to use flash_attention
            attn_mask=mask,
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, config: BaseModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    @torch.compile
    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


@torch.compile
def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
