# SmolTTS: a text-to-speech laboratory

This repo is a personal laboratory for training autoregressive text-audio models.

Assume everything will change; quality right now is pretty mid. Will get better.

## Using pretrained models

### smoltts_v0

A distillation of [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) to the RQ Transformer architecture. Released at 70M and 150M scale.

For MLX inference on Apple Silicon, you'll need a working Python installation. See the `mlx_inference` folder for setup docs!

```bash
# tl;dr
uvx --from smoltts_mlx smoltts-server
```

Candle.rs docs coming soon.

## Using datasets

As of Feb 2025, this project currently uses the [Mimi](https://huggingface.co/kyutai/mimi) pretrained codec by Kyutai, due to its low framerate (12.5Hz), high compression ratio, and streaming support.

### Synthetic data

[projectgutenberg-kokoro_v1-mimi](jkeisling/projectgutenberg-kokoro_v1-mimi):

- ~5500 hours of synthetic audio generated with [Kokoro v1](https://huggingface.co/hexgrad/Kokoro-82M) for US and UK English.
- 3 million utterances of sentences from Project Gutenberg, mostly 3-15s. 3.29GB compressed with Mimi.
- 11 speakers.

### Mimi re-encodings of standard datasets

For convenience, we serialize popular open TTS benchmark datasets in Mimi, to directly have training targets and compress the filesize by ~500x:

- [LibriTTS-R](https://huggingface.co/datasets/jkeisling/libritts-r-mimi) encoded with [Mimi](https://huggingface.co/kyutai/mimi) codec. ~460 hours of data.

## Pretraining a model

### Workspace setup

Unfortunately, HuggingFace Datasets using audio columns require librosa, which has a hard Python 3.9 dependency for inexplicable reasons.
If you are not creating a new dataset using raw audio instead of Mimi codes, please feel free to ignore this.

Please use [uv](https://docs.astral.sh/uv/).

```bash
# If you are not making new audio datasets, feel free to use a sane Python version instead
uv sync
uv pip install -e .
```

Create a `.env` file and add:

```bash
HUGGINGFACE_TOKEN=sk-placeholder
```

For the dataset and init, see `data_pipeline/README.md`.

### RQ Transformer

This architecture is most popularly used as the neural codec seq2seq backbone for:

- [Fish Speech TTS](https://github.com/fishaudio/fish-speech) (in their [paper](https://arxiv.org/html/2411.01156v2#S3) as "DualAR" or dual-autoregressive)
- Kyutai's [Moshi](https://github.com/kyutai-labs/moshi) model early in pretraining before adaptation to duplex audio.

Models trained here will be compatible with my DualAR [fish-speech.rs](https://github.com/EndlessReform/fish-speech.rs/blob/main/README.md) inference engine.
