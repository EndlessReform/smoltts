# Dataset utils

> [!INFO]
> Yes, I know, this isn't Meta AI-tier data engineering. Sue me.

Here's what you need to bootstrap the training setup:

## Model config

### Byte-level tokenizer

If you're training a _byte-level_ model from scratch, in the CLI, first navigate to the project root.
Then create an `inits` folder for the model config.
The name of your init can be arbitrary; we'll just assume you're using `smoltts_byte_kokoro`:

```bash
# Name can be arbitrary
mkdir -p inits/smoltts_byte_kokoro
```

Then let's create a byte-level HuggingFace tokenizer:

```bash
# From project root
uv run data_pipeline/scripts/create_bytelevel_init.py --out-dir inits/smoltts_byte_kokoro
```

Finally, copy your model config:

```bash
# Substitute model config as desired
cp sample_model_sizes/smoltts_byte_60m_wte.json ./inits/smoltts_byte_kokoro/config.json
```

### BPE

If you're using a real BPE tokenizer, run the `create_smoltts_init` notebook: Convert your LM base and tokenizer to DualAR format.

## Audio dataset creation

### LibriTTS-R dataset

- `encode_libritts.py` to encode, then
- `upload_libritts.ipynb` to upload if you feel this to be necessary.

Thanks to HF Datasets streaming, does not require _persisting_ all ~100-200GB of audio to your hard drive, just _downloading_ it. **Skip this step** if using [pre-encoded LibriTTS-R](https://huggingface.co/datasets/jkeisling/libritts-r-mimi).

### Synthetic data with Kokoro

See a different repo

### i love...EMILIA

## Tokenizing your data

Your dataset must contain the following columns:

- `text_normalized`: String
- `codes`: Tensor of Mimi codes
- `speaker_id`: string, name of speaker (e.g. "alloy")

Create your config (ideally version-controlled in `data_pipeline/scripts/audio_tokenizer_configs`). Then, for example:

```bash
uv run data_pipeline/scripts/chatml_tokenize_dataset.py \
    -c ./data_pipeline/scripts/audio_tokenizer_configs/project_gutenberg_v2.1.json \
    -o ./datasets/byte-tokenized-pg-kokoro_v1
```

For legacy BPE tokenization, use`tokenize_libritts.ipynb`: Tokenize, ChatML format, and pack by speaker index
