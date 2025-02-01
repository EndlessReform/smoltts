from argparse import ArgumentParser
from dotenv import load_dotenv
from datasets import load_dataset, load_from_disk
import json
import os
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer
from typing import Dict, Optional, List


from data_pipeline.utils.prompt import PromptEncoder, TokenizationConfig


class Config(BaseModel):
    frame_rate: float = Field(default=12.5)
    max_sample_secs: float = Field(default=15.0)
    tokenizer_path: str
    dataset_id: Optional[str] = Field(default=None)
    dataset_path: Optional[str] = Field(default=None)
    speaker_names: Optional[List[str]] = Field(default=None)
    default_sysprompt: Optional[str] = Field(default="Speak out the provided text.")
    is_bytelevel: bool = Field(default=False)


def tts_tokenize_row(
    row: Dict,
    prompt_encoder: PromptEncoder,
    dataset_config: Config,
    speaker_map: Optional[Dict[str, int]] = None,
):
    if speaker_map is not None and "speaker_id" in row:
        sysprompt_text = f"<|speaker:{speaker_map[row['speaker_id']]}"
    elif dataset_config.default_sysprompt is not None:
        sysprompt_text = dataset_config.default_sysprompt
    else:
        sysprompt_text = ""

    system_line = prompt_encoder.encode_text_turn(role="system", content=sysprompt_text)
    user_line = prompt_encoder.encode_text_turn(
        role="user",
        content=row["text_normalized"].encode("utf-8").decode("latin-1")
        if dataset_config.is_bytelevel
        else row["text_normalized"],
        add_generation_prompt=True,
    )
    assistant_line = prompt_encoder.encode_vq(row["codes"])

    ground_truth = torch.cat([system_line, user_line, assistant_line], dim=1)
    tokens = ground_truth[:, :-1].clone()
    labels = ground_truth[:, 1:].clone()

    text_only_length = system_line.size(1) + user_line.size(1) - 1
    labels[1:, :text_only_length] = -100
    # Mask out <|im_end|> and newline
    labels[1:, -2:] = -100

    return {
        "tokens": tokens,
        "labels": labels,
        "audio_length": row["codes"].size(-1) * dataset_config.frame_rate,
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

    print(f"Filtering rows above {dataset_config.max_sample_secs}s")
    dataset = dataset.filter(
        lambda row: row["codes"].size(-1)
        <= dataset_config.frame_rate * dataset_config.max_sample_secs,
        num_proc=NUM_PROC,
    )

    tokenizer = AutoTokenizer.from_pretrained(dataset_config.tokenizer_path)
    tokenizer.use_default_system_prompt = False
    tokenization_config = TokenizationConfig()

    prompt_encoder = PromptEncoder(tokenizer, tokenization_config)
    speaker_ids = (
        {value: index for index, value in enumerate(dataset_config.speaker_names)}
        if dataset_config.speaker_names is not None
        else None
    )

    print("Tokenizing dataset")
    dataset = dataset.map(
        lambda row: tts_tokenize_row(row, prompt_encoder, dataset_config, speaker_ids),
        remove_columns="codes",
        num_proc=NUM_PROC,
    )

    dataset.save_to_disk(args.out_path)


if __name__ == "__main__":
    main()
