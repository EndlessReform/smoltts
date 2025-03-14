import mlx.core as mx
import tokenizers
from tokenizers import Tokenizer
from typing import Optional, Tuple

from smoltts_mlx.lm.config import ModelType
from smoltts_mlx.lm.rq_transformer import RQTransformer


class FishPromptEncoder:
    tokenizer: Tokenizer
    depth: int
    model_type: ModelType

    def __init__(
        self,
        tokenizer: Tokenizer,
        model_type: ModelType,
        semantic_offset: int,
        num_codebooks: int = 8,
        duplicate_code_0: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.depth = num_codebooks if duplicate_code_0 else num_codebooks - 1
        self.semantic_offset = semantic_offset

    @classmethod
    def from_model(cls, tokenizer: Tokenizer, model: RQTransformer):
        return cls(
            tokenizer,
            num_codebooks=model.config.num_codebooks,
            model_type=model.model_type,
            semantic_offset=model.token_config.semantic_start_id,
            duplicate_code_0=dc0
            if (dc0 := model.config.duplicate_code_0) is not None
            else True,
        )

    def tokenize_text(self, text: str) -> mx.array:
        turn_codes: tokenizers.Encoding = self.tokenizer.encode(
            text, add_special_tokens=True
        )
        tokens = mx.array(turn_codes.ids, dtype=mx.uint32)[mx.newaxis, :]
        zeros = mx.zeros([self.depth, tokens.shape[-1]], dtype=mx.uint32)
        return mx.concat([tokens, zeros], axis=0)

    def encode_text_turn(self, role: str, content: Optional[str] = None) -> mx.array:
        content_suffix = f"{content}<|im_end|>" if content is not None else ""
        turn_string = f"<|im_start|>{role}\n{content_suffix}"
        return self.tokenize_text(turn_string)

    def encode_vq(self, codes: mx.array) -> mx.array:
        if codes.ndim != 2:
            raise ValueError("Must be single batch")

        semantic_line = (codes[0, :] + self.semantic_offset)[mx.newaxis, :]
        lower_start = codes.shape[0] - self.depth
        lower_codes = codes[lower_start:, :]
        vq_block = mx.concat([semantic_line, lower_codes])
        im_end = self.tokenize_text("<|im_end|>\n")
        block = mx.concat([vq_block, im_end], axis=1)
        return block


class CSMPromptEncoder:
    tokenizer: Tokenizer

    def __init__(
        self,
        tokenizer: Tokenizer,
        num_codebooks: int = 32,
    ):
        self.tokenizer = tokenizer
        self.depth = num_codebooks

    def tokenize_text(self, text: str) -> Tuple[mx.array, mx.array]:
        text_codes: tokenizers.Encoding = self.tokenizer.encode(
            text, add_special_tokens=True
        )
        tokens = mx.array(text_codes.ids, dtype=mx.int64)[:, mx.newaxis]
        text_frame = mx.concat(
            [mx.zeros([tokens.shape[0], self.depth], dtype=mx.int64), tokens], axis=-1
        )
        text_frame_mask = text_frame != 0
        return (text_frame, text_frame_mask)

    def tokenize_audio(self, mimi_codes: mx.array):
        assert mimi_codes.ndim == 2
        mimi_codes = mimi_codes.transpose(0, 1)
        eos_frame = mx.zeros([mimi_codes.shape[0], 1])
        # TODO is this right?
        audio_frame = mx.concat([mimi_codes, eos_frame], axis=0)
        audio_frame = mx.concat(
            [audio_frame, mx.zeros([audio_frame.shape[0], 1])], axis=1
        )
        audio_frame_mask = audio_frame != 0
        return (audio_frame, audio_frame_mask)
