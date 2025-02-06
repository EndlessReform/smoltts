from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from modeling.model.rq_transformer import RQTransformer
from train.config import TrainingConfig


class TrainingState(NamedTuple):
    model: RQTransformer
    optimizer: Optional[Optimizer]
    scheduler: Optional[LRScheduler]
    start_epoch: int
    global_step: int


class CheckpointManager:
    def __init__(self, base_directory: str, keep_last_n: int = 5):
        self.base_dir = Path(base_directory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        print(f"Checkpoint directory for this run: {self.run_dir}")

    def load_checkpoint(
        self, checkpoint_path: str, config: TrainingConfig, device: torch.device
    ) -> TrainingState:
        """Load a checkpoint and return initialized training state"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
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

        # Load model with original architecture but override weights
        model = RQTransformer.from_pretrained(
            config.init_folder,
            load_weights=False,
        )
        model_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(model_state_dict)

        # Move model to device and dtype
        model = model.to(device)
        model = model.to(torch.bfloat16)

        return TrainingState(
            model=model,
            optimizer=None,  # Caller will reinit if needed
            scheduler=None,  # Caller will reinit if needed
            start_epoch=checkpoint["epoch"],
            global_step=checkpoint["global_step"],
        )

    def init_model(self, config: TrainingConfig, device: torch.device) -> TrainingState:
        """Create a fresh model and training state"""
        model = RQTransformer.from_pretrained(
            config.init_folder, load_weights=config.use_pretrained
        )
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {num_params}")

        model = model.to(device)
        model = model.to(torch.bfloat16)

        print(f"Model max_seq_len: {model.max_seq_len}")

        return TrainingState(
            model=model, optimizer=None, scheduler=None, start_epoch=0, global_step=0
        )

    def save(self, state: TrainingState, config: TrainingConfig) -> None:
        """Save current training state"""
        if state.global_step == 0:
            print("Skipping step 0")
            return None

        state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in state.model.state_dict().items()
        }

        checkpoint_path = self.run_dir / f"step_{state.global_step}.pt"
        torch.save(
            {
                "epoch": state.start_epoch,
                "global_step": state.global_step,
                "model_state_dict": state_dict,
                "optimizer_state_dict": (
                    state.optimizer.state_dict() if state.optimizer else None
                ),
                "scheduler_state_dict": (
                    state.scheduler.state_dict() if state.scheduler else None
                ),
                "config": config.model_dump(),
            },
            checkpoint_path,
        )

        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints from current run"""
        files = sorted(self.run_dir.glob("step_*.pt"))
        if len(files) > self.keep_last_n:
            for f in files[: -self.keep_last_n]:
                f.unlink()
