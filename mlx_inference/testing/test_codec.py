from huggingface_hub import hf_hub_download
import mlx.core as mx
import time
from datasets import load_dataset
import numpy as np

from mlx_inference.codec.mimi import load_mimi
from mlx_inference.io.wav import pcm_to_wav_bytes


def main():
    dataset = load_dataset("jkeisling/libritts-r-mimi")
    dataset = dataset.with_format("numpy")
    arr = mx.array(dataset["dev.clean"][10]["codes"])
    test_input = arr[mx.newaxis, :, :]

    model = load_mimi()
    print("Model loaded")

    start_time = time.time()

    decoded = model.decode(test_input, None)
    mx.eval(decoded)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Done")
    print(f"Decoded shape: {decoded.shape}")
    print(f"Elapsed time: {(elapsed_time * 1000):.3f} ms")
    wav_bytes = pcm_to_wav_bytes(np.array(decoded))
    with open("output.wav", "wb") as f:
        f.write(wav_bytes)

    reference = np.load("final.npy")
    wav_bytes = pcm_to_wav_bytes(np.array(reference))
    with open("output_from_tbase.wav", "wb") as f:
        f.write(wav_bytes)


if __name__ == "__main__":
    main()
