from typing import List, Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from train.config import TrainingConfig


def partition_params(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Split params into decay/no-decay groups"""
    weight_decay_params, no_decay_params = [], []
    for name, param in sorted(model.named_parameters()):
        if param.requires_grad:  # Only include trainable params
            if ".bias" in name or "norm.weight" in name or ".embeddings." in name:
                no_decay_params.append(param)
            else:
                weight_decay_params.append(param)

    print(f"Weight decay params: {len(weight_decay_params)}")
    print(f"No decay params: {len(no_decay_params)}")
    return weight_decay_params, no_decay_params


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> AdamW:
    """Create optimizer with proper weight decay grouping"""
    weight_decay_params, no_decay_params = partition_params(model)

    return AdamW(
        [
            {"params": weight_decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )


def get_lr_scheduler(
    optimizer: AdamW, config: TrainingConfig, warmup_start_step: int = 0
) -> LambdaLR:
    """Creates scheduler with linear warmup"""

    def lr_lambda(current_step: int):
        relative_step = current_step - warmup_start_step
        if relative_step < config.lr_warmup_steps:
            # Linear decay from lr_start to learning_rate
            progress = float(relative_step) / float(max(1, config.lr_warmup_steps))
            return config.lr_start / config.learning_rate * (1.0 - progress) + progress
        return 1.0  # Return to base learning_rate

    return LambdaLR(optimizer, lr_lambda)


def setup_training(
    model: nn.Module,
    config: TrainingConfig,
    global_step: int = 0,
) -> Tuple[AdamW, LambdaLR]:
    """One-shot creation of optimizer and scheduler"""
    optimizer = create_optimizer(model, config)
    scheduler = get_lr_scheduler(optimizer, config, global_step)

    if global_step > 0:
        # Initialize scheduler with current step
        scheduler.last_epoch = (
            global_step - 1
        )  # -1 because scheduler steps once on first call

    return optimizer, scheduler
