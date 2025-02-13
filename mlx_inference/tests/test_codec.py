import mlx.core as mx
from datasets import load_dataset
import numpy as np

from smoltts_mlx.codec.mimi import load_mimi
from smoltts_mlx.io.wav import pcm_to_wav_bytes
from smoltts_mlx.lm.cache import make_prompt_cache


def main():
    dataset = load_dataset("jkeisling/libritts-r-mimi")
    dataset = dataset.with_format("numpy")
    arr = mx.array(dataset["dev.clean"][10]["codes"])
    test_input = arr[mx.newaxis, :, :]

    model = load_mimi()
    print("Model loaded")

    # start_time = time.time()

    # dont worry about 1: from the full TTS, it's audio-only here
    quantized = model.quantizer.decode(test_input)

    embeddings = model.upsample(quantized)
    transformed = model.decoder_transformer(embeddings)
    mx.eval(transformed)
    # decoded = model.decode(test_input, None)
    # mx.eval(decoded)
    print(f"TRANSFORMED: {transformed.shape}")

    all_pcm_conv_frame = []
    # upsample doubles the frame rate, so we need to match the actual streaming
    for frame in mx.split(
        transformed, axis=-2, indices_or_sections=transformed.shape[-2] // 2
    ):
        step = model.decoder.step(frame)
        if step is not None:
            # out = model.decoder(frame)
            all_pcm_conv_frame.append(mx.swapaxes(step, 1, 2))
        else:
            print("Skipping step")

    model.decoder.reset()

    decoded = mx.concat(all_pcm_conv_frame, axis=-1)

    # end_time = time.time()
    # elapsed_time = end_time - start_time

    print("Done")
    print(f"Decoded shape: {decoded.shape}")
    # print(f"Elapsed time: {(elapsed_time * 1000):.3f} ms")
    wav_bytes = pcm_to_wav_bytes(np.array(decoded))
    with open("output_conv.wav", "wb") as f:
        f.write(wav_bytes)

    print("Testing TRANSFORMER + RVQ + UPSAMPLE")
    frames = mx.split(test_input, axis=-1, indices_or_sections=test_input.shape[-1])
    print(frames[0].shape)
    quantized_incremental = [model.quantizer.decode(arr) for arr in frames]
    upsampled_embeddings = [model.upsample(x) for x in quantized_incremental]
    # embeddings = model.upsample(quantized_incremental)
    cache = make_prompt_cache(model.decoder_transformer)
    all_transformer_out = []
    # for frame in mx.split(
    #     embeddings, axis=-2, indices_or_sections=embeddings.shape[-2] // 2
    # ):
    for frame in upsampled_embeddings:
        emb = model.decoder_transformer(frame, cache=cache)
        all_transformer_out.append(emb)
    transformed_incremental = mx.concat(all_transformer_out, axis=-2)
    decoded_t = mx.swapaxes(model.decoder(transformed_incremental), 1, 2)

    # reference = np.load("final.npy")
    wav_bytes = pcm_to_wav_bytes(np.array(decoded_t))
    with open("output_t_incremental.wav", "wb") as f:
        f.write(wav_bytes)


if __name__ == "__main__":
    main()
