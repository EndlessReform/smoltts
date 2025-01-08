from datasets import load_dataset, Audio
import datasets
from transformers import MimiModel
from torch.nn.utils.rnn import pad_sequence
import torch
import math
import argparse
from tqdm import tqdm
from pathlib import Path


def get_target_length(arr: torch.Tensor) -> int:
    return math.ceil(arr.size(-1) / (SAMPLING_RATE / 12.5))


def batch_wav_encoder(batch_dict):
    # Each batch_dict has "audio" with a list of samples
    batch = batch_dict["audio"]

    LONG_AUDIO_THRESHOLD = 15 * SAMPLING_RATE  # 15 seconds
    regular_batch = []
    long_batch = []

    # Split short vs. long
    for sample in batch:
        if sample["array"].size(-1) > LONG_AUDIO_THRESHOLD:
            long_batch.append(sample)
        else:
            regular_batch.append(sample)

    all_outputs = []

    # Encode "regular" items
    if regular_batch:
        target_lengths = [get_target_length(s["array"]) for s in regular_batch]
        padded_batch = pad_sequence(
            [s["array"] for s in regular_batch], batch_first=True
        ).unsqueeze(
            1
        )  # (batch, 1, time)

        with torch.no_grad():
            enc_out_cuda = model.encode(padded_batch.to("cuda"))
        enc_out = enc_out_cuda.audio_codes[:, 0:8, :].clone().cpu()
        del enc_out_cuda
        torch.cuda.empty_cache()

        chunked = torch.unbind(enc_out, dim=0)
        outputs = [t[:, :l] for t, l in zip(chunked, target_lengths)]
        all_outputs.extend(outputs)

    # Encode "long" items in smaller mini-batches
    if long_batch:
        LONG_BATCH_SIZE = 4
        for i in range(0, len(long_batch), LONG_BATCH_SIZE):
            mini = long_batch[i : i + LONG_BATCH_SIZE]
            target_lengths = [get_target_length(s["array"]) for s in mini]
            padded_batch = pad_sequence(
                [s["array"] for s in mini], batch_first=True
            ).unsqueeze(1)

            with torch.no_grad():
                enc_out_cuda = model.encode(padded_batch.to("cuda"))
            enc_out = enc_out_cuda.audio_codes[:, 0:8, :].clone().cpu()
            del enc_out_cuda
            torch.cuda.empty_cache()

            chunked = torch.unbind(enc_out, dim=0)
            outputs = [t[:, :l] for t, l in zip(chunked, target_lengths)]
            all_outputs.extend(outputs)

    return {"codes": all_outputs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="encoded_libritts",
        help="Directory where each split folder will be saved",
    )
    args = parser.parse_args()

    global SAMPLING_RATE, model
    SAMPLING_RATE = 24_000

    # Load model
    model = MimiModel.from_pretrained("kyutai/mimi").to("cuda")

    # The splits we want to process
    all_splits = [
        "dev.clean",
        "test.clean",
        "train.clean.100",
        "train.clean.360",
    ]

    # Load the source dataset with streaming
    dataset_dict = load_dataset("mythicinfinity/libritts_r", "clean", streaming=True)

    output_dir = Path(args.output)

    for split in all_splits:
        print(f"\nProcessing {split}...")

        # We'll stream the dataset, with the audio column cast to the correct sample rate
        streamed_split = dataset_dict[split]
        streamed_split = streamed_split.cast_column(
            "audio", Audio(sampling_rate=SAMPLING_RATE)
        )
        streamed_split = streamed_split.with_format("torch")

        # We'll accumulate encoded rows in memory
        # (If you truly can't fit them all, you'd reintroduce sharding.)
        encoded_rows = []
        print("Encoding in batches...")

        for batch_out in tqdm(
            streamed_split.map(
                batch_wav_encoder,
                batched=True,
                batch_size=24,
                remove_columns=["audio"],
            ),
            desc=f"Encoding {split}",
        ):
            encoded_rows.append(batch_out)

        new_data = datasets.Dataset.from_list(encoded_rows)

        # Save to disk in a subfolder named after the split
        split_folder = output_dir / split
        print(f"Saving {split} to {split_folder}...")
        split_folder.mkdir(parents=True, exist_ok=True)
        new_data.save_to_disk(str(split_folder))

        print(f"Finished {split}")

    print("\nAll splits processed. Done!")


if __name__ == "__main__":
    main()
