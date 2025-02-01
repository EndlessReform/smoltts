from argparse import ArgumentParser
from dotenv import load_dotenv
from dataclasses import dataclass
from datasets import load_dataset, load_from_disk
import json
import os
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer
from typing import Dict, Optional, List, Literal


from data_pipeline.utils.prompt import PromptEncoder, TokenizationConfig


class TokenizationStrategy(BaseModel):
    tokenizer_path: str
    strategy: Literal["bpe", "bytelevel", "phoneme", "hybrid"]


class AudioConfig(BaseModel):
    frame_rate: float = Field(default=12.5)
    max_sample_secs: float = Field(default=15.0)


class SpeakerStrategy(BaseModel):
    strategy: Literal["id_token", "fixed", "omit"]
    speaker_names: Optional[List[str]] = Field(default=None)
    default_sysprompt: Optional[str] = Field(default=None)


class PackingStrategy(BaseModel):
    max_sequence_length: int = Field(default=768)
    max_items_per_pack: int = Field(default=5)
    window_size: int = Field(default=1600)


class Config(BaseModel):
    dataset_id: Optional[str] = Field(default=None)
    dataset_path: Optional[str] = Field(default=None)
    tokenization: TokenizationStrategy
    speaker: SpeakerStrategy
    audio: AudioConfig
    packing: Optional[PackingStrategy] = Field(default=None)


@dataclass
class SyspromptEntry:
    tokens: torch.Tensor
    labels: torch.Tensor


class SyspromptEncoder:
    default_sysprompt: Optional[SyspromptEntry] = None
    speaker_cache: Optional[Dict[str, SyspromptEntry]] = None

    def __init__(self, dataset_config: Config, prompt_encoder: PromptEncoder):
        self.dataset_config = dataset_config
        self.prompt_encoder = PromptEncoder
        if dataset_config.speaker.default_sysprompt is not None:
            # One single sysprompt
            raw_prompt = prompt_encoder.encode_text_turn(
                role="system",
                content=dataset_config.speaker.default_sysprompt,
                add_generation_prompt=False,
            )
            self.default_sysprompt = self.causal_shift(raw_prompt)
        elif dataset_config.speaker.speaker_names is not None:
            # Precompute speaker prompt cache if we have a known small subset
            self.speaker_cache = {
                speaker_name: self.causal_shift(
                    prompt_encoder.encode_text_turn(
                        role="system",
                        content=f"<|speaker:{id}|>",
                        add_generation_prompt=False,
                    ),
                )
                for id, speaker_name in enumerate(dataset_config.speaker.speaker_names)
            }

    def causal_shift(self, ground_truth: torch.Tensor) -> SyspromptEntry:
        tokens = ground_truth[:, :-1].clone()
        labels = ground_truth[:, 1:].clone()
        labels[1:, :] = -100
        return SyspromptEntry(tokens=tokens, labels=labels)

    def get_sysprompt_length(self, row: Dict) -> int:
        if self.default_sysprompt is not None:
            # Fixed
            return self.default_sysprompt.tokens.size(-1)
        elif self.speaker_cache is not None:
            # Speaker ID from known set
            return self.speaker_cache[row["speaker_id"]].tokens.size(-1)
        else:
            # TODO handle arbitrary token length
            return 0

    def add_sysprompt(self, row: Dict):
        if self.dataset_config.speaker.strategy == "omit":
            return {}
        else:
            if self.default_sysprompt is not None:
                speaker_entry = self.default_sysprompt
            elif self.speaker_cache is not None:
                speaker_entry = self.speaker_cache[row["speaker_id"]]
            else:
                raise ValueError(
                    f"Must have default syprompt or IDs, current strategy: {self.dataset_config.speaker.strategy}"
                )

            return {
                "tokens": torch.cat([speaker_entry.tokens, row["tokens"]], dim=1),
                "labels": torch.cat([speaker_entry.labels, row["labels"]], dim=1),
            }


def tts_tokenize_row(
    row: Dict,
    prompt_encoder: PromptEncoder,
    dataset_config: Config,
):
    """
    NOTE: unlike the notebook, this does NOT handle speaker prompt
    """
    user_line = prompt_encoder.encode_text_turn(
        role="user",
        content=row["text_normalized"].encode("utf-8").decode("latin-1")
        if dataset_config.tokenization.strategy == "bpe"
        else row["text_normalized"],
        add_generation_prompt=True,
    )
    assistant_line = prompt_encoder.encode_vq(row["codes"])

    ground_truth = torch.cat([user_line, assistant_line], dim=1)
    tokens = ground_truth[:, :-1].clone()
    labels = ground_truth[:, 1:].clone()

    text_only_length = user_line.size(1) - 1
    labels[1:, :text_only_length] = -100
    # Mask out <|im_end|> and newline
    labels[1:, -2:] = -100

    return {
        "tokens": tokens,
        "labels": labels,
        "audio_length": row["codes"].size(-1) * dataset_config.audio.frame_rate,
    }


parser = ArgumentParser(
    description="Tokenize Mimi-encoded dataset for final consumption"
)
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument(
    "-o", "--out-path", type=str, required=True, help="Local path of dataset output"
)


# TODO configure this
NUM_PROC = 12


def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = json.load(f)
        dataset_config = Config(**config_dict)

    load_dotenv()
    if dataset_config.dataset_path:
        dataset = load_from_disk(dataset_config.dataset_path)
    elif dataset_config.dataset_id:
        dataset = load_dataset(
            dataset_config.dataset_id, token=os.getenv("HUGGINGFACE_TOKEN")
        )
    else:
        raise ValueError("Neither dataset_id nor dataset_path specified in config!")

    print("Loaded dataset")
    dataset = dataset.with_format("torch")

    tokenizer = AutoTokenizer.from_pretrained(
        dataset_config.tokenization.tokenizer_path
    )
    tokenizer.use_default_system_prompt = False
    tokenization_config = TokenizationConfig()
    prompt_encoder = PromptEncoder(tokenizer, tokenization_config)
    sysprompt_encoder = SyspromptEncoder(
        dataset_config=dataset_config, prompt_encoder=prompt_encoder
    )

    print(f"Filtering rows above {dataset_config.audio.max_sample_secs}s")
    dataset = dataset.filter(
        lambda row: row["codes"].size(-1)
        <= dataset_config.audio.frame_rate * dataset_config.audio.max_sample_secs,
        num_proc=NUM_PROC,
    )

    print("Tokenizing dataset")
    dataset = dataset.map(
        lambda row: tts_tokenize_row(row, prompt_encoder, dataset_config),
        remove_columns="codes",
        num_proc=NUM_PROC,
    )
    print("Adding system prompt")
    dataset = dataset.map(sysprompt_encoder.add_sysprompt, num_proc=NUM_PROC)

    dataset.save_to_disk(args.out_path)


if __name__ == "__main__":
    main()
