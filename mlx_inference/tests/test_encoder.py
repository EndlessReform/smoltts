import mlx.core as mx
# import numpy as np

from smoltts_mlx.codec.mimi import load_mimi
# from smoltts_mlx.io.wav import pcm_to_wav_bytes
# from smoltts_mlx.lm.cache import make_prompt_cache


def main():
    # arr = mx.array(dataset["dev.clean"][10]["codes"])
    # test_input = arr[mx.newaxis, :, :]
    # 2s of audio
    test_input = mx.zeros(48_000)[mx.newaxis, :, mx.newaxis]

    model = load_mimi()
    print("Model loaded")

    # dont worry about 1: from the full TTS, it's audio-only here
    embedded = model.encoder(test_input)
    transformed = model.encoder_transformer(embedded)
    downsampled = model.downsample(transformed)
    codes = model.quantizer.encode(downsampled)
    mx.save("out.npy", mx.swapaxes(codes, 0, 1))


if __name__ == "__main__":
    main()
