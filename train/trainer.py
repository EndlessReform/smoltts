from dataclasses import dataclass
from einops import rearrange
from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import List
import wandb

from modeling.model.rq_transformer import RQTransformer
from train.config import TrainingConfig
from train.data import collate_fn
from train.state import TrainingState, CheckpointManager


@dataclass
class TrainStepOutput:
    loss: torch.Tensor
    base_loss: float
    semantic_loss: float
    lr: float


def compute_losses(
    outputs, labels: torch.Tensor, per_codebook_loss: bool = False
) -> tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Compute base and semantic losses, plus individual codebook losses"""
    # Base loss computation remains the same
    base_loss = F.cross_entropy(
        outputs.token_logits.view(-1, outputs.token_logits.size(-1)),
        labels[:, 0, :].reshape(-1),
        ignore_index=-100,
    )

    # Compute individual codebook losses
    n_codebooks = labels.shape[1] - 1  # Subtract 1 for the base tokens
    if per_codebook_loss:
        codebook_losses = []

        for i in range(n_codebooks):
            # Reshape logits and labels for current codebook
            current_logits = outputs.codebook_logits[:, :, i, :]  # [batch, seq, vocab]
            current_labels = labels[:, i + 1, :]  # [batch, seq]

            loss = F.cross_entropy(
                current_logits.reshape(-1, current_logits.size(-1)),
                current_labels.reshape(-1),
                ignore_index=-100,
            )
            codebook_losses.append(loss.item())
    else:
        codebook_losses = []

    # Compute total semantic loss (same as before, just using einops)
    codebook_logits = rearrange(outputs.codebook_logits, "b s n d -> (b s n) d")
    codebook_labels = rearrange(labels[:, 1:, :], "b n s -> (b s n)")
    semantic_loss = F.cross_entropy(codebook_logits, codebook_labels, ignore_index=-100)

    return base_loss, semantic_loss, codebook_losses


def train_step(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    accumulate_steps: int = 1,  # New parameter for loss scaling
) -> TrainStepOutput:
    """
    Executes a forward pass and backward pass (with loss scaling for gradient accumulation).
    Does NOT perform optimizer.step() or gradient clipping here.
    """
    tokens = batch["tokens"].to(device)
    labels = batch["labels"].to(device)
    pad_mask = batch["pad_mask"].to(device)

    outputs = model(inp=tokens, key_padding_mask=pad_mask)
    base_loss, semantic_loss, _ = compute_losses(outputs, labels)

    # Compute total loss and scale it for accumulation
    total_loss = base_loss + semantic_loss
    scaled_loss = total_loss / accumulate_steps
    scaled_loss.backward()

    # Return the unscaled losses for logging purposes
    return TrainStepOutput(
        loss=total_loss,
        base_loss=base_loss.item(),
        semantic_loss=semantic_loss.item(),
        lr=0.0,  # Will be updated in the training loop after optimizer.step()
    )


def validate(
    model: torch.nn.Module, val_loader: DataLoader, device: torch.device
) -> dict:
    """Run validation"""
    model.eval()
    total_loss = total_base_loss = total_semantic_loss = num_batches = 0
    total_codebook_losses = None

    with torch.no_grad():
        for batch in val_loader:
            tokens = batch["tokens"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(tokens)
            base_loss, semantic_loss, codebook_losses = compute_losses(outputs, labels)
            loss = base_loss + semantic_loss

            # Initialize total_codebook_losses on first batch
            if total_codebook_losses is None:
                total_codebook_losses = [0.0] * len(codebook_losses)

            # Accumulate losses
            total_loss += loss.item()
            total_base_loss += base_loss.item()
            total_semantic_loss += semantic_loss.item()
            total_codebook_losses = [
                total + current
                for total, current in zip(total_codebook_losses, codebook_losses)
            ]
            num_batches += 1

            del tokens, labels, outputs, base_loss, semantic_loss, loss
            torch.cuda.empty_cache()

    model.train()
    return {
        "loss": total_loss / num_batches,
        "base_loss": total_base_loss / num_batches,
        "semantic_loss": total_semantic_loss / num_batches,
        "codebook_losses": [loss / num_batches for loss in total_codebook_losses],
    }


def create_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    config: TrainingConfig,
    pad_id: int,
    duplicate_code_0: bool = True,
) -> tuple[DataLoader, DataLoader]:
    pad_collate_fn = partial(
        collate_fn, semantic_pad_id=pad_id, duplicate_code_0=duplicate_code_0
    )

    """Create train and validation dataloaders"""
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train(
    model: RQTransformer,
    train_ds: Dataset,
    val_ds: Dataset,
    config: TrainingConfig,
    device: torch.device,
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[torch.optim.lr_scheduler.LRScheduler],
    checkpoint_manager: CheckpointManager,
    start_epoch: int = 0,
    global_step: int = 0,
):
    pad_id = model.tokenizer.pad_token_id
    if config.use_wandb:
        wandb.init(project=config.project_name, resume="allow")
        wandb.config.update(config.model_dump())

    train_loader, val_loader = create_dataloaders(
        train_ds,
        val_ds,
        config,
        pad_id=pad_id,
        duplicate_code_0=model.config.duplicate_code_0,
    )

    # Initialize accumulation counter and zero the gradients initially.
    accumulation_counter = 0
    for optimizer in optimizers:
        optimizer.zero_grad()

    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            # Forward and backward pass (loss scaled inside train_step)
            step_output = train_step(
                model, batch, device=device, accumulate_steps=config.accumulate_steps
            )

            accumulation_counter += 1

            # Only perform an optimizer step when enough gradients have accumulated.
            if accumulation_counter == config.accumulate_steps:
                # Optionally clip gradients
                if config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )
                for optimizer in optimizers:
                    optimizer.step()

                for scheduler in schedulers:
                    scheduler.step()

                for optimizer in optimizers:
                    optimizer.zero_grad()
                accumulation_counter = 0
                torch.cuda.empty_cache()

            # Get current learning rate (even if not updated, it stays the same)
            current_lr_display = []
            for scheduler in schedulers:
                current_lr_display.append(f"{scheduler.get_last_lr()[0]:.2e}")

            if config.use_wandb:
                metrics = {
                    "train/loss": float(step_output.loss),
                    "train/base_loss": float(step_output.base_loss),
                    "train/semantic_loss": float(step_output.semantic_loss),
                    "train/learning_rate": current_lr_display,
                    "epoch": epoch,
                }
                wandb.log(metrics, step=global_step)

            progress_bar.set_postfix(
                loss=f"lm={step_output.base_loss:.4f},codes={step_output.semantic_loss:.4f}",
                lr=current_lr_display,
            )

            if (
                global_step % config.val_every_n_steps == 0
                and config.use_wandb
                and global_step != 0
            ):
                val_metrics = validate(model, val_loader, device)
                wandb.log(
                    {
                        "val/loss": float(val_metrics["loss"]),
                        "val/base_loss": float(val_metrics["base_loss"]),
                        "val/semantic_loss": float(val_metrics["semantic_loss"]),
                        **{
                            f"val/codebook_{i + 1}_loss": loss
                            for i, loss in enumerate(val_metrics["codebook_losses"])
                        },
                    },
                    step=global_step,
                )

            if global_step % config.save_every_n_steps == 0:
                checkpoint_manager.save(
                    TrainingState(
                        model=model,
                        optimizer=optimizers,
                        scheduler=schedulers,
                        start_epoch=epoch,
                        global_step=global_step,
                    ),
                    config,
                )

            global_step += 1

    checkpoint_manager.save(
        TrainingState(
            model=model,
            optimizer=optimizers,
            scheduler=schedulers,
            start_epoch=config.max_epochs - 1,
            global_step=global_step,
        ),
        config,
    )
