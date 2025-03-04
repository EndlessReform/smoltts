from datasets import load_dataset
from data_pipeline.utils.codec import MimiCodec
from dotenv import load_dotenv
import os
import shutil
import torch
from torchaudio.transforms import Resample

# from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from typing import Dict

DATASET_SAMPLING_RATE = 24_000

downsample_16k = Resample(orig_freq=DATASET_SAMPLING_RATE)


# def encode_row(row: Dict):
#     audio = row["mp3"]["array"]
#     downsampled = downsample_16k(audio)
#     inputs = emb_feature_extractor(
#         downsampled, padding=True, return_tensors="pt", sampling_rate=16_000
#     )
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         embeddings = emb_model(**inputs).embeddings
#     del inputs
#     embeddings_cpu = embeddings.cpu()
#     del embeddings

#     encoded = codec.encode(audio.unsqueeze(0))
#     encoded_cpu = encoded.cpu()

#     torch.cuda.empty_cache()
#     return {"codes": encoded_cpu, "speaker_emb": embeddings_cpu}


def main():
    load_dotenv()

    # TODO parameterize in CLI
    paths = [f"Emilia/EN/EN-B00{i:04d}.tar" for i in range(5)]
    dataset = load_dataset(
        "amphion/Emilia-Dataset",
        data_files=paths,
        split="train",
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )
    dataset = dataset.with_format("pt")

    codec = MimiCodec()
    # please for the love of god don't run this on a laptop
    # device = "cuda"

    # emb_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    #     "microsoft/wavlm-base-plus-sv"
    # )
    # emb_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
    # emb_model = emb_model.to(device).eval()

    def encode_batch(batch: Dict):
        audio = [a["array"] for a in batch["mp3"]]
        encoded = codec.encode_batch(audio)
        return {"codes": encoded}

    # fuck it
    test_ds = dataset
    test_ds.map(encode_batch, batched=True, batch_size=24)
    dataset_dir = os.path.expanduser(
        "~/.cache/huggingface/datasets/amphion___emilia-dataset"
    )

    # Say goodbye
    try:
        shutil.rmtree(dataset_dir, ignore_errors=True)  # IGNORE ERRORS: NO MERCY
        print(f"💥 Nuked: {dataset_dir}")
    except Exception as e:  # Just in case something dares to resist
        print(f"🔥 Failed to nuke {dataset_dir}: {e}")


if __name__ == "__main__":
    main()
