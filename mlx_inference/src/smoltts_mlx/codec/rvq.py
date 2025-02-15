import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel
from typing import Optional


class RVQConfig(BaseModel):
    codebook_size: int = 2048
    codebook_dim: int = 256
    num_quantizers: int = 32
    num_semantic_quantizers: int = 1
    frame_rate: float = 12.5
    hidden_dim: int = 512


@mx.compile
def cdist(x: mx.array, y: mx.array):
    x1_square_norms = mx.sum(x**2, axis=-1, keepdims=True)
    x2_square_norms = mx.swapaxes(mx.sum(y**2, axis=-1, keepdims=True), 2, 1)

    dot_products = mx.matmul(x, mx.swapaxes(y, 2, 1))
    dists_sq = x1_square_norms + x2_square_norms - 2 * dot_products
    return dists_sq.sqrt()


class MimiEuclideanCodebook(nn.Module):
    """
    Codebook with Euclidean distance.
    """

    def __init__(self, config: RVQConfig, epsilon: float = 1e-5):
        super().__init__()
        self.embed_sum = mx.zeros([config.codebook_size, config.codebook_dim])
        self.codebook_size = config.codebook_size
        self.epsilon = epsilon
        # This does nothing at inference, but we need it so the keys line up
        self.initialized = mx.array([True], dtype=mx.float32)
        self.cluster_usage = mx.ones(config.codebook_size)
        self._embed = None

    @property
    def embed(self) -> mx.array:
        if self._embed is None:
            self._embed = (
                self.embed_sum
                / mx.maximum(self.cluster_usage, self.epsilon)[:, mx.newaxis]
            )
        return self._embed

    def decode(self, embed_ind) -> mx.array:
        quantize = self.embed[embed_ind, :]
        return quantize

    def quantize(self, x: mx.array) -> mx.array:
        dists = cdist(x[mx.newaxis, :], self.embed[mx.newaxis, :])[0]
        embed_ind = dists.argmin(axis=-1)
        return embed_ind

    def encode(self, x: mx.array) -> mx.array:
        shape = x.shape
        x = x.reshape((-1, shape[-1]))
        embed_ind = self.quantize(x)
        embed_ind = embed_ind.reshape(shape[:-1])
        return embed_ind


class MimiVectorQuantization(nn.Module):
    def __init__(self, config: RVQConfig):
        super().__init__()
        self.codebook = MimiEuclideanCodebook(config)

    def encode(self, x: mx.array) -> mx.array:
        embed_in = self.codebook.encode(x)
        return embed_in

    def decode(self, embed_ind: mx.array) -> mx.array:
        quantize = self.codebook.decode(embed_ind)
        out = mx.transpose(quantize, (0, 2, 1))
        return out


class MimiResidualVectorQuantizer(nn.Module):
    def __init__(self, config: RVQConfig, num_quantizers: Optional[int] = None):
        super().__init__()
        self.num_quantizers = (
            num_quantizers if num_quantizers is not None else config.num_quantizers
        )
        self.layers = [
            MimiVectorQuantization(config) for _ in range(self.num_quantizers)
        ]

        self.input_proj = nn.Conv1d(
            config.hidden_dim, config.codebook_dim, 1, bias=False
        )
        self.output_proj = nn.Conv1d(
            config.codebook_dim, config.hidden_dim, 1, bias=False
        )

    def encode(
        self, embeddings: mx.array, num_quantizers: Optional[int] = None
    ) -> mx.array:
        embeddings = self.input_proj(embeddings)
        num_quantizers = (
            num_quantizers if num_quantizers is not None else self.num_quantizers
        )

        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - mx.swapaxes(quantized, 1, 2)
            all_indices.append(indices)

        out_indices = mx.stack(all_indices)
        return out_indices

    def decode(self, codes: mx.array) -> mx.array:
        quantized_out = mx.array(0.0)
        codes = mx.swapaxes(codes, 0, 1)
        for i, indices in enumerate(codes):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized

        # (bsz, dim, seqlen) to dim first
        quantized_out = mx.swapaxes(quantized_out, 1, 2)
        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out)

        return quantized_out


class MimiSplitResidualVectorQuantizer(nn.Module):
    def __init__(self, config: RVQConfig):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.max_num_quantizers = config.num_quantizers

        self.num_semantic_quantizers = config.num_semantic_quantizers
        self.num_acoustic_quantizers = (
            config.num_quantizers - config.num_semantic_quantizers
        )

        self.semantic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            config, self.num_semantic_quantizers
        )
        self.acoustic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            config, self.num_acoustic_quantizers
        )

    def encode(
        self, embeddings: mx.array, num_quantizers: Optional[int] = None
    ) -> mx.array:
        num_quantizers = (
            self.max_num_quantizers if num_quantizers is None else num_quantizers
        )

        if num_quantizers > self.max_num_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be lower than the total number of quantizers {self.max_num_quantizers}, but is currently {num_quantizers}."
            )

        if num_quantizers < self.num_semantic_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be higher than the number of semantic quantizers {self.num_semantic_quantizers}, but is currently {num_quantizers}."
            )

        codes = self.semantic_residual_vector_quantizer.encode(embeddings)
        if num_quantizers > self.num_semantic_quantizers:
            acoustic_codes = self.acoustic_residual_vector_quantizer.encode(
                embeddings, num_quantizers=num_quantizers - self.num_semantic_quantizers
            )
            codes = mx.concat([codes, acoustic_codes], axis=0)

        return codes

    def decode(self, codes: mx.array):
        quantized_out = self.semantic_residual_vector_quantizer.decode(
            codes[:, : self.num_semantic_quantizers]
        )
        quantized_out += self.acoustic_residual_vector_quantizer.decode(
            codes[:, self.num_semantic_quantizers :]
        )
        return quantized_out
