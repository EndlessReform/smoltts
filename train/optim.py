from muon import Muon
from typing import List, Tuple, Optional
from torch.optim import AdamW, Optimizer
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
            if (
                ".bias" in name
                or "norm.weight" in name
                or ".embeddings." in name
                or ".fast_output" in name
            ):
                no_decay_params.append(param)
            else:
                weight_decay_params.append(param)

    print(f"Weight decay params: {len(weight_decay_params)}")
    print(f"No decay params: {len(no_decay_params)}")
    return weight_decay_params, no_decay_params


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> Tuple[AdamW, Optional[Muon]]:
    """Create optimizer with proper weight decay grouping"""
    weight_decay_params, no_decay_params = partition_params(model)
    if config.optimizer.muon is None:
        return (
            AdamW(
                [
                    {
                        "params": weight_decay_params,
                        "weight_decay": config.optimizer.weight_decay,
                    },
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=config.optimizer.learning_rate,
                betas=config.optimizer.betas,
                eps=config.optimizer.eps,
            ),
            None,
        )
    else:
        muon_params = [p for p in weight_decay_params if p.ndim >= 2]
        adam_wd_params = [p for p in weight_decay_params if p.ndim < 2]
        return (
            AdamW(
                [
                    {
                        "params": adam_wd_params,
                        "weight_decay": config.optimizer.weight_decay,
                    },
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=config.optimizer.learning_rate,
                betas=config.optimizer.betas,
                eps=config.optimizer.eps,
            ),
            Muon(
                muon_params,
                lr=config.optimizer.muon.learning_rate,
                momentum=config.optimizer.muon.momentum,
            ),
        )


def get_lr_scheduler(
    optimizer: Optimizer, config: TrainingConfig, warmup_start_step: int = 0
) -> LambdaLR:
    """Creates scheduler with linear warmup"""

    def lr_lambda(current_step: int):
        relative_step = current_step - warmup_start_step
        if relative_step < config.optimizer.lr_warmup_steps:
            # Linear decay from lr_start to learning_rate
            progress = float(relative_step) / float(
                max(1, config.optimizer.lr_warmup_steps)
            )
            return (
                config.optimizer.lr_start
                / config.optimizer.learning_rate
                * (1.0 - progress)
                + progress
            )
        return 1.0  # Return to base learning_rate

    return LambdaLR(optimizer, lr_lambda)


def setup_training(
    model: nn.Module,
    config: TrainingConfig,
    global_step: int = 0,
) -> Tuple[List[Optimizer], List[LambdaLR]]:
    """One-shot creation of optimizer and scheduler"""
    adamw_optim, maybe_muon_optim = create_optimizer(model, config)
    adamw_scheduler = get_lr_scheduler(adamw_optim, config, global_step)

    if global_step > 0:
        # Initialize scheduler with current step
        adamw_scheduler.last_epoch = (
            global_step - 1
        )  # -1 because scheduler steps once on first call

    if maybe_muon_optim is None:
        return ([adamw_optim], [adamw_scheduler])
    else:
        muon_scheduler = get_lr_scheduler(maybe_muon_optim, config, global_step)
        if global_step > 0:
            # Initialize scheduler with current step
            muon_scheduler.last_epoch = (
                global_step - 1
            )  # -1 because scheduler steps once on first call

        return ([adamw_optim, maybe_muon_optim], [adamw_scheduler, muon_scheduler])
