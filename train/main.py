import argparse
import os
import torch
from train.config import load_config
from train.data import load_splits
from train.optim import setup_training
from train.state import CheckpointManager
from train.trainer import train
import torch.distributed as dist

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


def main():
    # Requiring config now
    config = load_config(args.config)
    train_ds, val_ds = load_splits(config.dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize state management
    checkpoint_manager = CheckpointManager(config.checkpoint_path)

    # Load or create model and training state
    if args.checkpoint:
        state = checkpoint_manager.load_checkpoint(args.checkpoint, config, device)
    else:
        state = checkpoint_manager.init_model(config, device)

    # Setup optimizer and scheduler
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    optimizers, schedulers = setup_training(state.model, config, state.global_step)

    # Environment setup
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    print(f"Dropout: {state.model.layers[0].attention.dropout}")

    # Start training
    train(
        state.model,
        train_ds,
        val_ds,
        config,
        device,
        optimizers,
        schedulers,
        checkpoint_manager,
        state.start_epoch,
        state.global_step,
    )


if __name__ == "__main__":
    main()
