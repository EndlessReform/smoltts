from argparse import ArgumentParser
import json
import os
from pydantic import BaseModel, Field
from tokenizers import Tokenizer, models, decoders
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

CHATML_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}" 

class Config(BaseModel):
    codebook_size: int = Field(default=2048)


def get_blank_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    trainer = BpeTrainer(vocab_size=256, special_tokens=[])

    # Create actual bytes
    byte_data = [bytes([i]) for i in range(256)]
    # Preserve the actual byte values in strings
    byte_strings = [b.decode('latin-1') for b in byte_data]

    # "Train" the tokenizer
    tokenizer.train_from_iterator(byte_strings, trainer=trainer)
    tokenizer.pre_tokenizer = None
    tokenizer.normalizer = None
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer

def add_special_tokens(tokenizer: Tokenizer, config: Config) -> Tokenizer:
    semantic_tokens = [f"<|semantic:{i}|>" for i in range(config.codebook_size)]
    control_tokens = [
        "system",
        "user",
        "assistant",
        "<|british|>",
        "<|american|>",
        "<|male|>",
        "<|female|>",
        "<|unknown|>",
        "<|endoftext|>",
        "<|voice|>",
        "<|semantic|>",
        "<|pad|>",
        "<|epad|>",
        "<|im_start|>",
        "<|im_end|>",
    ]

    speaker_id_tokens = [f"<|speaker:{i}|>" for i in range(64 - len(control_tokens))]
    charset = [*control_tokens, *speaker_id_tokens, *semantic_tokens]

    tokenizer.add_special_tokens(charset)

    return tokenizer

parser = ArgumentParser(description="Create BPE tokenizer for Kokoro-style bounded speaker models")
parser.add_argument(
    "--config-file",
    type=str,
)
parser.add_argument(
    "--out-dir", "-o",
    required=True,
    help="Directory where tokenizer will be saved to"
)

def main():
    args = parser.parse_args()
    if args.config_file is not None:
        with open(args.config_file_path, "r") as f:
            config_data = json.load(f)
        
        config = Config(**config_data)
    else:
        # Default is fine
        config = Config()

    base_tokenizer = get_blank_tokenizer()
    tokenizer = add_special_tokens(base_tokenizer, config)

    final_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|endoftext|>",
        unk_token="<|unknown|>",
        pad_token="<|pad|>",
        chat_template=CHATML_TEMPLATE
    )
    
    os.makedirs(args.out_dir, exist_ok=True)
    final_tokenizer.save_pretrained(args.out_dir)
    print("Saving complete!")

if __name__ == "__main__":
    main()