from datasets import load_from_disk, Dataset
from typing import Tuple
import torch


def load_splits(path: str, max_sequence_len: int = 768) -> Tuple[Dataset, Dataset]:
    """
    Returns (train, val) datasets
    """
    TEST_SIZE = 10_000
    print(f"Loading dataset from {path}")
    SEED = 108
    dataset = load_from_disk(path)
    dataset = dataset.with_format("torch")
    if isinstance(dataset, Dataset):
        dataset = dataset.train_test_split(test_size=TEST_SIZE)
    print(f"Keys: {dataset.keys()}")
    if "full" in (splits := list(dataset.keys())):
        dataset = dataset["full"].shuffle()
        split_dataset = dataset.train_test_split(test_size=TEST_SIZE)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
    elif "val" in splits:
        train_dataset = dataset["train"].shuffle(SEED)
        val_dataset = dataset["val"]
    elif "test" in splits:
        train_dataset = dataset["train"].shuffle(SEED)
        val_dataset = dataset["test"]
    else:
        dataset.shuffle(SEED)
        split_dataset = dataset["train"].train_test_split(test_size=TEST_SIZE)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]

    print(train_dataset.column_names)
    print(val_dataset.column_names)
    return train_dataset, val_dataset


def collate_fn(
    batch,
    max_seq_len: int,
    semantic_pad_id: int,
    duplicate_code_0: bool = True,
    codebook_size: int = 8,
):
    height = codebook_size + (1 if duplicate_code_0 else 0)
    # Just silently eat outliers here, assume upstream salvaged the bulk of them
    rows = [
        item["ground_truth"]
        for item in batch
        if item["ground_truth"].shape[1] <= max_seq_len
    ]
    max_input_len = max(row.shape[1] - 1 for row in rows)

    B = len(rows)
    # We'll create padded arrays:
    tokens = torch.full((B, height, max_input_len), 0, dtype=torch.long)  # 2=some <PAD>
    tokens[:, 0, :] = semantic_pad_id
    labels = torch.full(
        (B, height, max_input_len), -100, dtype=torch.long
    )  # default is ignore_index

    pad_mask = torch.ones(B, max_input_len)

    for i, row in enumerate(rows):
        seq_len = row.shape[1] - 1
        tokens[i, :, :seq_len] = row[:, :-1].clone()

        label = row[:, 1:]
        text_only_mask = label[1:, :] == 0
        label[1:, :][text_only_mask] = -100
        labels[i, :, :seq_len] = label

        pad_mask[i, :seq_len] = False

    return {"tokens": tokens, "labels": labels, "pad_mask": pad_mask}
