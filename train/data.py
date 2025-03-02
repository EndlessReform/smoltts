from datasets import load_from_disk, Dataset
from typing import Tuple
import torch


def load_splits(path: str, max_sequence_len: int = 768) -> Tuple[Dataset, Dataset]:
    """
    Returns (train, val) datasets
    """
    print(f"Loading dataset from {path}")
    dataset = load_from_disk(path)
    dataset = dataset.with_format("torch")
    print(f"Keys: {dataset.keys()}")
    if "full" in (splits := list(dataset.keys())):
        dataset = dataset["full"].shuffle()
        split_dataset = dataset.train_test_split(test_size=5000)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
    elif "val" in splits:
        train_dataset = dataset["train"].shuffle(42)
        val_dataset = dataset["val"]
    else:
        dataset.shuffle(42)
        split_dataset = dataset["train"].train_test_split(test_size=5000)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]

    print(train_dataset.column_names)
    print(val_dataset.column_names)
    return train_dataset, val_dataset


def collate_fn(
    batch, semantic_pad_id: int, duplicate_code_0: bool = True, codebook_size: int = 8
):
    """
    batch is a list of dicts: each dict has "tokens" shape [9, T],
    and "labels" shape [9, T].
    We pad them into [B, 9, T_max].
    """
    # TODO handle >8 codebooks
    height = codebook_size + (1 if duplicate_code_0 else 0)
    max_input_len = max(item["tokens"].shape[1] for item in batch)

    B = len(batch)
    # We'll create padded arrays:
    tokens = torch.full((B, height, max_input_len), 0, dtype=torch.long)  # 2=some <PAD>
    tokens[:, 0, :] = semantic_pad_id
    labels = torch.full(
        (B, height, max_input_len), -100, dtype=torch.long
    )  # default is ignore_index

    pad_mask = torch.ones(B, max_input_len)

    for i, item in enumerate(batch):
        seq_len = item["tokens"].shape[1]
        tokens[i, :, :seq_len] = item["tokens"]
        labels[i, :, :seq_len] = item["labels"][:, :seq_len]
        pad_mask[i, :seq_len] = False

    return {"tokens": tokens, "labels": labels, "pad_mask": pad_mask}
