import argparse
from datasets import load_from_disk, Dataset
from dual_ar.model.dual_ar import DualARTransformer
from einops import rearrange
import os
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import wandb


class TrainingConfig(BaseModel):
    # Core paths and identifiers
    project_name: str = "ljspeech_train"
    checkpoint_path: str = "checkpoints"
    model_path: str = "pretrained_model"

    # Training params
    batch_size: int = 8
    max_epochs: int = 10
    num_workers: int = 4
    gradient_clip: float = 1.0

    # Optimizer settings
    learning_rate: float = 1e-4
    lr_start: float = 1e-3
    lr_warmup_steps: int = 3000
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-5

    # Validation & Checkpointing
    val_every_n_steps: int = 100
    save_every_n_steps: int = 500

    # Model/Data params
    max_sequence_length: int = 512  # Much smaller than original 4096 for LJSpeech
    use_bf16: bool = True
    use_wandb: bool = False
    use_pretrained: bool = True


def load_config(path: str) -> TrainingConfig:
    import json

    with open(path) as f:
        config_dict = json.load(f)
    return TrainingConfig(**config_dict)


def load_splits(
    path="../dataset/tokenized_libritts",
) -> Tuple[Dataset, Dataset]:
    # TODO stop hard-coding this once we experiment with encodings
    print(f"Loading dataset from {path}")
    dataset = load_from_disk(path)
    if "full" in list(dataset.keys()):
        dataset["full"].shuffle(42)
        split_dataset = dataset["full"].train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
    else:
        train_dataset = dataset["train"].shuffle(42)
        val_dataset = dataset["val"]
    return train_dataset, val_dataset


def collate_fn(batch):
    """
    batch is a list of dicts: each dict has "tokens" shape [9, T],
    and "labels" shape [9, T].
    We pad them into [B, 9, T_max].
    """
    # if not hasattr(collate_fn, "printed_debug"):
    #     print("First batch debug:")
    #     print(" - tokens shape:", batch[0]["tokens"].shape)
    #     print(" - labels shape:", batch[0]["labels"].shape)
    #     print("Sample row=0, first 20 labels:", batch[0]["labels"][0, :20])
    #     print("Sample row=1, first 20 labels:", batch[0]["labels"][1, :20])
    #     collate_fn.printed_debug = True

    max_input_len = max(item["tokens"].shape[1] for item in batch)

    B = len(batch)
    # We'll create padded arrays:
    tokens = torch.full((B, 9, max_input_len), 0, dtype=torch.long)  # 2=some <PAD>
    tokens[:, 0, :] = 2
    labels = torch.full(
        (B, 9, max_input_len), -100, dtype=torch.long
    )  # default is ignore_index

    pad_mask = torch.ones(B, max_input_len)

    for i, item in enumerate(batch):
        seq_len = item["tokens"].shape[1]
        tokens[i, :, :seq_len] = item["tokens"]
        labels[i, :, :seq_len] = item["labels"][:, :seq_len]
        pad_mask[i, :seq_len] = False

    return {"tokens": tokens, "labels": labels, "pad_mask": pad_mask}


def get_lr_scheduler(optimizer, config: TrainingConfig, warmup_start_step: int = 0):
    """
    Creates a learning rate scheduler that starts at lr_start and decays
    linearly to learning_rate over warmup_steps. If warmup_start_step is provided,
    starts warmup from that step.
    """

    def lr_lambda(current_step: int):
        relative_step = current_step - warmup_start_step
        if relative_step < config.lr_warmup_steps:
            # Linear decay from lr_start to learning_rate
            progress = float(relative_step) / float(max(1, config.lr_warmup_steps))
            return config.lr_start / config.learning_rate * (1.0 - progress) + progress
        return 1.0  # Return to base learning_rate

    return LambdaLR(optimizer, lr_lambda)


def freeze_base_trunk(model: DualARTransformer):
    """Freeze the base LM trunk while keeping codebook and fast transformer components trainable.
    Must be called AFTER model initialization but BEFORE optimizer setup."""

    # Freeze base transformer components
    model.embeddings.requires_grad_(False)
    for layer in model.layers:
        for param in layer.parameters():
            param.requires_grad = False
    model.norm.requires_grad_(False)

    # Everything else (fast transformer, codebook components) stays trainable by default

    # Verify freezing
    trainable_modules = []
    frozen_modules = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_modules.append(name)
        else:
            frozen_modules.append(name)

    print("Trainable modules:")
    for name in trainable_modules:
        print(f"  {name}")
    print("\nFrozen modules:")
    for name in frozen_modules:
        print(f"  {name}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")


def train_loop(
    model,
    train_ds,
    val_ds,
    config: TrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,  # New parameter
    scheduler: torch.optim.lr_scheduler.LRScheduler,  # New parameter
    start_epoch: int = 0,
    global_step: int = 0,
):
    # Init wandb
    if config.use_wandb:
        wandb.init(project=config.project_name, resume="allow")
        wandb.config.update(config.model_dump())

    # Setup dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            tokens = batch["tokens"].to(device)
            labels = batch["labels"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            # Forward pass
            outputs = model(inp=tokens, key_padding_mask=pad_mask)

            # Loss calculation
            base_loss = F.cross_entropy(
                # Flatten to (bsz * seqlen, vocab_size)
                outputs.token_logits.view(-1, outputs.token_logits.size(-1)),
                # bsz * seqlen
                labels[:, 0, :].reshape(-1),
                ignore_index=-100,
            )

            # (bsz * seqlen * n_codebooks, codebook_pred)
            codebook_logits = rearrange(outputs.codebook_logits, "b s n d -> (b s n) d")
            # (batch * sequence * n_codebooks)
            codebook_labels = rearrange(labels[:, 1:, :], "b n s -> (b s n)")

            semantic_loss = F.cross_entropy(
                codebook_logits, codebook_labels, ignore_index=-100
            )
            loss = base_loss + semantic_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()

            # Logging
            if config.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/base_loss": base_loss.item(),
                        "train/semantic_loss": semantic_loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[
                            0
                        ],  # Log the current LR
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            progress_bar.set_postfix(
                loss=f"lm={base_loss.item():.4f},codes={semantic_loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            # Validation
            if global_step % config.val_every_n_steps == 0 and config.use_wandb:
                val_metrics = validate(model, val_loader, device)
                wandb.log(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/base_loss": val_metrics["base_loss"],
                        "val/semantic_loss": val_metrics["semantic_loss"],
                    },
                    step=global_step,
                )

            # Save checkpoint
            if global_step % config.save_every_n_steps == 0:
                state_dict = {
                    k.replace("_orig_mod.", ""): v
                    for k, v in model.state_dict().items()
                }

                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),  # Add this line
                        "config": config.model_dump(),
                    },
                    f"{config.checkpoint_path}/step_{global_step}.pt",
                )

            global_step += 1

    print("FINAL SAVE")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),  # Add this line
            "config": config.model_dump(),
        },
        f"{config.checkpoint_path}/step_{global_step}.pt",
    )


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_base_loss = 0
    total_semantic_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            tokens = batch["tokens"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(tokens)

            base_loss = F.cross_entropy(
                outputs.token_logits.view(-1, outputs.token_logits.size(-1)),
                labels[:, 0].reshape(-1),
                ignore_index=-100,
            )

            codebook_labels = labels[:, 1 : 1 + model.config.num_codebooks].transpose(
                1, 2
            )
            semantic_loss = F.cross_entropy(
                outputs.codebook_logits.view(-1, outputs.codebook_logits.size(-1)),
                codebook_labels.reshape(-1),
                ignore_index=-100,
            )

            loss = base_loss + semantic_loss

            total_loss += loss.item()
            total_base_loss += base_loss.item()
            total_semantic_loss += semantic_loss.item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "base_loss": total_base_loss / num_batches,
        "semantic_loss": total_semantic_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file to resume from",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file to resume from",
    )
    args = parser.parse_args()

    config = load_config(
        args.config if args.config is not None else "../config/librispeech.json"
    )
    print("Loading datasets")
    train_ds, val_ds = load_splits()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, scheduler and training state
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        checkpoint_config = TrainingConfig(**checkpoint["config"])

        # Check for config mismatches that would affect optimizer/scheduler
        optimizer_keys = ["learning_rate", "weight_decay", "betas", "eps"]
        scheduler_keys = ["lr_start", "lr_warmup_steps"]

        optimizer_changed = any(
            getattr(config, key) != getattr(checkpoint_config, key)
            for key in optimizer_keys
        )
        scheduler_changed = any(
            getattr(config, key) != getattr(checkpoint_config, key)
            for key in scheduler_keys
        )

        if optimizer_changed or scheduler_changed:
            print("Detected changes in optimization parameters:")
            if optimizer_changed:
                print("Optimizer changes:")
                for key in optimizer_keys:
                    old_val = getattr(checkpoint_config, key)
                    new_val = getattr(config, key)
                    if old_val != new_val:
                        print(f"  {key}: {old_val} -> {new_val}")
            if scheduler_changed:
                print("Scheduler changes:")
                for key in scheduler_keys:
                    old_val = getattr(checkpoint_config, key)
                    new_val = getattr(config, key)
                    if old_val != new_val:
                        print(f"  {key}: {old_val} -> {new_val}")
            print("Will reinitialize optimizer and scheduler with new settings")

        # Load model with original architecture/tokenizer but override weights
        model = DualARTransformer.from_pretrained(
            "../checkpoints/smoltts_init",
            load_weights=False,  # Don't load original weights since we'll override
        )
        model_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(model_state_dict)
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]

    else:
        # Original initialization path
        model = DualARTransformer.from_pretrained(
            "../checkpoints/smoltts_init", load_weights=config.use_pretrained
        )
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {num_params}")
        start_epoch = 0
        global_step = 0

    # Common model setup
    model = model.to(device)
    model = model.to(torch.bfloat16)

    # Optimizer setup (common to both paths)
    # Optimizer setup (common to both paths)
    weight_decay_params, no_decay_params = [], []
    for name, param in sorted(
        model.named_parameters()
    ):  # Sort to ensure consistent ordering
        if param.requires_grad:  # Only include trainable params
            if ".bias" in name or "norm.weight" in name or ".embeddings." in name:
                no_decay_params.append(param)
            else:
                weight_decay_params.append(param)

    print(f"Weight decay params: {len(weight_decay_params)}")
    print(f"No decay params: {len(no_decay_params)}")

    optimizer = AdamW(
        [
            {"params": weight_decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )

    warmup_start = global_step if args.checkpoint else 0
    scheduler = get_lr_scheduler(optimizer, config, warmup_start)
    if args.checkpoint:
        # Initialize scheduler with current step
        scheduler.last_epoch = (
            global_step - 1
        )  # -1 because scheduler will step once on first call

    # Final common setup
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    print(f"Dropout: {model.layers[0].attention.dropout}")
    # model = torch.compile(model)

    # Start training
    train_loop(
        model,
        train_ds,
        val_ds,
        config,
        device,
        optimizer,
        scheduler,
        start_epoch,
        global_step,
    )


if __name__ == "__main__":
    main()
