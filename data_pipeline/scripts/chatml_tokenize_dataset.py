from argparse import ArgumentParser
from dotenv import load_dotenv
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
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
    duplicate_code_0: Optional[bool] = True


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


class SyspromptEncoder:
    default_sysprompt: Optional[torch.Tensor] = None
    speaker_cache: Optional[Dict[str, torch.Tensor]] = None

    def __init__(self, dataset_config: Config, prompt_encoder: PromptEncoder):
        self.dataset_config = dataset_config
        self.prompt_encoder = PromptEncoder
        if dataset_config.speaker.default_sysprompt is not None:
            # One single sysprompt
            self.default_sysprompt = prompt_encoder.encode_text_turn(
                role="system",
                content=dataset_config.speaker.default_sysprompt,
                add_generation_prompt=False,
            )
        elif dataset_config.speaker.speaker_names is not None:
            # Precompute speaker prompt cache if we have a known small subset
            self.speaker_cache = {
                speaker_name: prompt_encoder.encode_text_turn(
                    role="system",
                    content=f"<|speaker:{id}|>",
                    add_generation_prompt=False,
                )
                for id, speaker_name in enumerate(dataset_config.speaker.speaker_names)
            }

    def get_sysprompt_length(self, speaker_id: str) -> int:
        if self.default_sysprompt is not None:
            # Fixed
            return self.default_sysprompt.size(-1)
        elif self.speaker_cache is not None:
            # Speaker ID from known set
            return self.speaker_cache[speaker_id].size(-1)
        else:
            # TODO handle arbitrary token length
            return 0

    def add_sysprompt(
        self, ground_truth: torch.Tensor, speaker_id: str
    ) -> torch.Tensor:
        if self.dataset_config.speaker.strategy == "omit":
            return ground_truth
        else:
            if self.default_sysprompt is not None:
                speaker_entry = self.default_sysprompt
            elif self.speaker_cache is not None:
                speaker_entry = self.speaker_cache[speaker_id]
            else:
                raise ValueError(
                    f"Must have default syprompt or IDs, current strategy: {self.dataset_config.speaker.strategy}"
                )

            return torch.cat([speaker_entry, ground_truth], dim=1)


def tts_tokenize_row(
    row: Dict,
    prompt_encoder: PromptEncoder,
    dataset_config: Config,
):
    """
    NOTE: unlike the notebook, this does NOT handle
    - Speaker prompt
    - Causal shift
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

    return {
        "ground_truth": ground_truth.clone(),
    }


def causal_shift_row(row):
    tokens = row["ground_truth"][:, :-1].clone()
    labels = row["ground_truth"][:, 1:].clone()

    text_only_mask = labels[1:, :] == 0
    labels[1:, :][text_only_mask] = -100
    return {"tokens": tokens, "labels": labels}


def pack_utterances(batch: Dict, sysprompt_encoder: SyspromptEncoder):
    # Group utterances by speaker
    speakers = {}

    for speaker, tokens in zip(batch["speaker_id"], batch["ground_truth"]):
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(tokens)

    # Greedy packing per speaker (First-fit decreasing)
    for speaker in speakers:
        speakers[speaker].sort(key=lambda x: x.size(-1), reverse=True)

    packed_bins = []
    packed_ids = []
    for speaker, utterances in speakers.items():
        sysprompt_length = sysprompt_encoder.get_sysprompt_length(speaker_id=speaker)
        bins = []
        for utterance in utterances:
            placed = False
            for i in range(len(bins)):
                if (
                    bins[i].size(-1) + utterance.size(-1) + sysprompt_length
                    <= sysprompt_encoder.dataset_config.packing.max_sequence_length
                ):
                    bins[i] = torch.cat([bins[i], utterance], dim=1)
                    placed = True
                    break
            if not placed:
                bins.append(utterance)

        packed_bins += bins
        packed_ids += [speaker] * len(bins)

    packed_bins = [
        sysprompt_encoder.add_sysprompt(seq, speaker_id)
        for seq, speaker_id in zip(packed_bins, packed_ids)
    ]

    return {"ground_truth": packed_bins, "speaker_id": packed_ids}


parser = ArgumentParser(
    description="Tokenize Mimi-encoded dataset for final consumption"
)
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument(
    "-o", "--out-path", type=str, required=True, help="Local path of dataset output"
)
parser.add_argument("--shards", type=int)


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
    if "text" in dataset["train"].column_names:
        dataset = dataset.rename_column("text", "text_normalized")
    if "speaker" in dataset["train"].column_names:
        dataset = dataset.rename_column("speaker", "speaker_id")

    tokenizer = AutoTokenizer.from_pretrained(
        dataset_config.tokenization.tokenizer_path
    )
    tokenizer.use_default_system_prompt = False
    tokenization_config = TokenizationConfig(
        duplicate_code_0=dataset_config.tokenization.duplicate_code_0
    )
    prompt_encoder = PromptEncoder(tokenizer, tokenization_config)
    sysprompt_encoder = SyspromptEncoder(
        dataset_config=dataset_config, prompt_encoder=prompt_encoder
    )

    n_shards = args.shards if args.shards is not None else 1

    full_dataset = dataset
    completed = []
    for i in range(n_shards):
        dataset = full_dataset["train"].shard(n_shards, i)
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

        if dataset_config.packing is not None:
            print("Packing sequence")
            dataset = dataset.map(
                lambda row: pack_utterances(row, sysprompt_encoder),
                batched=True,
                batch_size=dataset_config.packing.window_size,
                num_proc=NUM_PROC,
                remove_columns=dataset.column_names,
            )

        completed.append(dataset)

    dataset = concatenate_datasets(completed)

    # print("Adding system prompt")
    # dataset = dataset.map(sysprompt_encoder.add_sysprompt, num_proc=NUM_PROC)
    # print("Causally shifting tokens, masking text-only")
    # dataset = dataset.map(
    #     causal_shift_row, num_proc=NUM_PROC, remove_columns=["ground_truth"]
    # )
    dataset = DatasetDict({"train": dataset})

    dataset.save_to_disk(args.out_path, max_shard_size="5GB")


if __name__ == "__main__":
    main()
