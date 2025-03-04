from pydantic import BaseModel, Field
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Optional, Union


class TokenizationConfig(BaseModel):
    num_codebooks: int = Field(default=8)
    acoustic_delay: int = Field(default=0)
    duplicate_code_0: Optional[bool] = Field(default=True)


class PromptEncoder:
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    num_codebooks: int
    trailing_im_end: torch.Tensor
    semantic_offset: int

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        config: TokenizationConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.semantic_offset = tokenizer.encode("<|semantic:0|>")[0]
        self.pad_id = tokenizer.encode("<|pad|>")[0]
        zero_buffer = [0] * (
            self.config.num_codebooks
            if self.config.duplicate_code_0
            else self.config.num_codebooks - 1
        )
        self.trailing_im_end = torch.tensor(
            [
                tokenizer.encode("<|im_end|>") + zero_buffer,
                tokenizer.encode("\n") + zero_buffer,
            ]
        ).T

    def get_lower_zeros(self, length: int) -> torch.Tensor:
        return torch.zeros(
            self.config.num_codebooks
            if self.config.duplicate_code_0
            else self.config.num_codebooks - 1,
            length,
            dtype=torch.long,
        )

    def tokenize_text(self, text: str) -> torch.Tensor:
        turn_codes = (
            self.tokenizer(text, return_tensors="pt").unsqueeze(0).to(torch.uint32)
        )
        zeros_mask = self.get_lower_zeros(turn_codes.size(-1))
        return torch.concat([turn_codes, zeros_mask], dim=0)

    def encode_text_turn(
        self, role: str, content: str, add_generation_prompt: bool = True
    ) -> torch.Tensor:
        baseline = self.tokenizer.apply_chat_template(
            [{"role": role, "content": content}],
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        zeros_mask = self.get_lower_zeros(baseline.size(-1))
        return torch.cat([baseline, zeros_mask], dim=0)

    def encode_vq(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 2:
            raise ValueError("Must be single batch")

        semantic_line = (codes[0, :] + self.semantic_offset).unsqueeze(0)
        lower_codes = codes if self.config.duplicate_code_0 else codes[1:, :]

        # TODO DO NOT MERGE, WRONG BAD
        if self.config.acoustic_delay != 0:
            semantic_suffix = torch.tensor(
                [self.pad_id] * self.config.acoustic_delay, dtype=torch.uint32
            ).unsqueeze(0)
            lower_codes_prefix = self.get_lower_zeros(self.config.acoustic_delay)
            semantic_line = torch.cat([semantic_line, semantic_suffix], dim=1)
            lower_codes = torch.cat([lower_codes_prefix, lower_codes], dim=1)

        vq_block = torch.cat([semantic_line, lower_codes])
        block = torch.cat([vq_block, self.trailing_im_end], dim=1)
        return block

    def encode_vq_corrupt(self, codes: torch.Tensor, dropout=0.2) -> torch.Tensor:
        """
        NO temporal delays or offsetting.

        Corrupts only the non-semantic codes.
        """
        if codes.ndim != 2:
            raise ValueError("Must be single batch!")

        semantic_line = (codes[0, :] + self.semantic_offset).unsqueeze(0)
        first_residual = codes[0, :].unsqueeze(0)
        remaining_codes = codes[1:, :]

        corrupt_mask = torch.rand_like(remaining_codes.float()) < dropout

        # TODO: parameterize for 1024-size codebook
        random_codes = torch.randint(
            low=1, high=2048, size=remaining_codes.shape, device=remaining_codes.device
        )

        corrupted_remaining_codes = torch.where(
            corrupt_mask, random_codes, remaining_codes
        )
        vq_block = torch.cat([semantic_line, first_residual, corrupted_remaining_codes])
        block = torch.cat([vq_block, self.trailing_im_end], dim=1)

        return block
