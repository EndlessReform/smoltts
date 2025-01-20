import mlx.core as mx
import tokenizers
from tokenizers import Tokenizer
from typing import Optional

from mlx_inference.model.config import ModelType
from mlx_inference.model.dual_ar import DualARTransformer


class PromptEncoder:
    tokenizer: Tokenizer
    num_codebooks: int
    model_type: ModelType

    def __init__(
        self, tokenizer: Tokenizer, model_type: ModelType, num_codebooks: int = 8
    ):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.num_codebooks = num_codebooks

    @classmethod
    def from_model(cls, tokenizer: Tokenizer, model: DualARTransformer):
        return cls(
            tokenizer,
            num_codebooks=model.config.num_codebooks,
            model_type=model.model_type,
        )

    def tokenize_text(self, text: str) -> mx.array:
        turn_codes: tokenizers.Encoding = self.tokenizer.encode(
            text, add_special_tokens=True
        )
        tokens = mx.array(turn_codes.ids, dtype=mx.uint32)[mx.newaxis, :]
        zeros = mx.zeros([self.num_codebooks, tokens.shape[-1]], dtype=mx.uint32)
        return mx.concat([tokens, zeros], axis=0)

    def encode_text_turn(self, role: str, content: Optional[str] = None) -> mx.array:
        content_suffix = f"{content}<|im_end|>" if content is not None else ""
        turn_string = f"<|im_start|>{role}\n{content_suffix}"
        return self.tokenize_text(turn_string)
