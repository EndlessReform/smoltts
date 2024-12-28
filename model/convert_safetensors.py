import sys
from safetensors.torch import save_file
import torch


def main():
    data = torch.load(sys.argv[1])
    model = data["model_state_dict"]
    save_file(model, "model.safetensors")


if __name__ == "__main__":
    main()
