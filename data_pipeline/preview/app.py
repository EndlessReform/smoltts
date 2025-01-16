from datasets import load_dataset, load_from_disk
from data_pipeline.utils.codec import MimiCodec
import gradio as gr
import json
import os
from pathlib import Path
from pydantic import BaseModel
import torch
from typing import List, Tuple


class DatasetConfig(BaseModel):
    dataset_path: Path
    sample_rate: int = 16000
    batch_size: int = 5
    split: str


def load_config(config_path: str = "decoder_config.json") -> DatasetConfig:
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
            # Brute force resolve relative path from script location
            config_data["dataset_path"] = os.path.abspath(
                os.path.join(os.path.dirname(__file__), config_data["dataset_path"])
            )
            return DatasetConfig(**config_data)
    except FileNotFoundError:
        config = DatasetConfig(
            dataset_path=Path("./data/my_dataset.hf"),
            sample_rate=24000,
            batch_size=5,
            split="full",
        )
        with open(config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
        print(f"Created default config at {config_path}")
        return config


# Rest of your gradio app
# Load config at startup
CONFIG = load_config()

# # Load dataset in streaming mode at startup
dataset = load_from_disk(
    dataset_path=str(CONFIG.dataset_path / CONFIG.split),
    keep_in_memory=False,
).shuffle()
dataset_iter = iter(dataset)
mimi_model = MimiCodec()


def get_random_samples(n: int = 5) -> List[Tuple[str, torch.Tensor]]:
    """Grab n random samples from our dataset"""
    n = n or CONFIG.batch_size
    samples = []
    for _ in range(n):
        sample = next(dataset_iter)
        samples.append((sample["text_normalized"], torch.tensor(sample["codes"])))
    return samples


def decode_and_display():
    """Get random samples and decode them"""
    samples = get_random_samples()
    outputs = []

    for text, codes in samples:
        # YOUR DECODE LOGIC HERE
        # Assuming codes is [8, seqlen] tensor of hierarchical codes
        # Return whatever format Gradio Audio widget expects

        # Placeholder for your decode logic
        fake_audio = mimi_model.decode(codes).squeeze(0)

        outputs.extend(
            [
                text,
                gr.Audio(
                    value=(CONFIG.sample_rate, fake_audio.numpy()),
                    type="numpy",
                ),
            ]
        )

    return outputs


with gr.Blocks() as demo:
    gr.Markdown(
        f"""
    ## Quick Mimi Validation
    Loading from: {CONFIG.dataset_path}
    """
    )

    with gr.Row():
        roll_btn = gr.Button("🎲 Roll Random Samples", variant="primary")

    # Create N rows of Text + Audio pairs
    display_rows = []
    for i in range(CONFIG.batch_size):
        with gr.Row():
            text = gr.Textbox(label=f"Text {i+1}", lines=2)
            audio = gr.Audio(label=f"Audio {i+1}")
            display_rows.extend([text, audio])

    roll_btn.click(fn=decode_and_display, outputs=display_rows)

# Launch with a decent sized queue for rapid checking
# demo.queue(max_size=10).launch()
demo.launch(server_name="0.0.0.0")
