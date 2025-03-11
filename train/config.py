import json
from pydantic import BaseModel
from typing import Tuple, Optional


class MuonSettings(BaseModel):
    learning_rate: float = 1e-2
    momentum: float = 0.95


class OptimizerSettings(BaseModel):
    learning_rate: float = 1e-4
    lr_start: float = 1e-3
    lr_warmup_steps: int = 3000
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-5
    muon: Optional[MuonSettings] = None


class TrainingConfig(BaseModel):
    # Core paths and identifiers
    project_name: str = "ljspeech_train"
    checkpoint_path: str = "checkpoints"
    model_path: str = "pretrained_model"
    dataset_path: str
    init_folder: str

    # Training params
    batch_size: int = 8
    max_epochs: int = 10
    num_workers: int = 4
    gradient_clip: float = 1.0
    accumulate_steps: int = 1

    # Optimizer settings
    optimizer: OptimizerSettings

    # Validation & Checkpointing
    val_every_n_steps: int = 100
    save_every_n_steps: int = 500

    # Model/Data params
    max_sequence_length: int = 896  # Much smaller than original 4096 for LJSpeech
    use_bf16: bool = True
    use_wandb: bool = False
    use_pretrained: bool = True
    compute_amortize_k: Optional[int] = None


def load_config(path: str) -> TrainingConfig:
    with open(path) as f:
        config_dict = json.load(f)
    return TrainingConfig(**config_dict)
