# Dataset notebooks

> [!INFO]
> Yes, I know, this isn't Meta AI-tier data engineering. Sue me.

Here's what you need to bootstrap the training setup:

## Model config

If you're training a _byte-level_ model, use:

```bash
# From project root
uv run data_pipeline/scripts/create_bytelevel_init.py --out-dir inits/smoltts_byte_kokoro
# Substtiute model size as desired
cp sample_model_sizes/smoltts_byte_60m_wte.json ./inits/smoltts_byte_kokoro/config.json
```

If you're using a real BPE tokenizer, run the `create_smoltts_init` notebook: Convert your LM base and tokenizer to DualAR format. 

## Audio dataset creation

LibriTTS-R dataset: `encode_libritts.py` to encode, then `upload_libritts.ipynb` to upload if you feel this to be necessary. Thanks to HF Datasets streaming, does not require _persisting_ all ~100-200GB of audio to your hard drive, just _downloading_ it. **Skip this step** if using [pre-encoded LibriTTS-R](https://huggingface.co/datasets/jkeisling/libritts-r-mimi).

## Final tokenization

tokenize_libritts.ipynb`: Tokenize, ChatML format, and pack by speaker index
