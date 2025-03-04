from datasets import load_dataset
from data_pipeline.utils.codec import MimiCodec
from dotenv import load_dotenv
import os
import shutil
import torch
from itertools import islice
from torchaudio.transforms import Resample

DATASET_SAMPLING_RATE = 24_000
CHUNK_SIZE = 5  # Tweak based on RAM

downsample_16k = Resample(orig_freq=DATASET_SAMPLING_RATE)


def chunked(iterable, size):
    """Yield successive chunks from iterable of given size."""
    it = iter(iterable)
    while chunk := list(islice(it, size)):  # Python 3.8+ (walrus op)
        yield chunk


def main():
    load_dotenv()

    codec = MimiCodec()
    dataset_dir_base = os.path.expanduser("~/local_datasets/emilia_chunks")

    os.makedirs(dataset_dir_base, exist_ok=True)

    for idx, chunk in enumerate(chunked(range(500), CHUNK_SIZE)):
        print(f"\n🟢 Processing chunk {idx + 1}/{(500 // CHUNK_SIZE) + 1}: {chunk}")

        paths = [f"Emilia/EN/EN-B00{i:04d}.tar" for i in chunk]

        print(f"📥 Downloading {len(paths)} files...")
        dataset = load_dataset(
            "amphion/Emilia-Dataset",
            data_files=paths,
            split="train",
            token=os.getenv("HUGGINGFACE_TOKEN"),
        )
        dataset = dataset.with_format("pt")

        def encode_batch(batch):
            audio = [a["array"] for a in batch["mp3"]]
            encoded = codec.encode_batch(audio)
            return {"codes": encoded}

        # Process & Save
        dataset.map(encode_batch, batched=True, batch_size=24)
        save_path = os.path.join(dataset_dir_base, f"chunk_{idx + 1}.pt")
        torch.save(dataset, save_path)
        print(f"💾 Saved chunk {idx + 1} to {save_path}")

        # Nuke cache
        dataset_dir = os.path.expanduser(
            "~/.cache/huggingface/datasets/amphion___emilia-dataset"
        )
        shutil.rmtree(dataset_dir, ignore_errors=True)
        print(f"🔥 Cleared cache for chunk {idx + 1}")

    print("✅ ALL CHUNKS PROCESSED.")


if __name__ == "__main__":
    main()
