## Dataset notebooks

> [!INFO]
> Yes, I know, this isn't Meta AI-tier data engineering. Sue me.

Here's what you need to bootstrap the training setup:

1. `create_smoltts_init`: Convert your LM base and tokenizer to DualAR format
2. LibriTTS-R dataset: `encode_libritts.py` to encode, then `upload_libritts.ipynb` to upload if you feel this to be necessary. Thanks to HF Datasets streaming, does not require _persisting_ all ~100-200GB of audio to your hard drive, just _downloading_ it. **Skip this step** if using [pre-encoded LibriTTS-R](https://huggingface.co/datasets/jkeisling/libritts-r-mimi).
3. `tokenize_libritts.ipynb`: Tokenize, ChatML format, and pack by speaker index
