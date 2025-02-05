import sys
from safetensors.torch import save_file
import torch


def main():
    data = torch.load(sys.argv[1])
    model = data["model_state_dict"]
    model = {key.replace("_orig_mod.", ""): value for key, value in model.items()}
    if model["fast_output.weight"].ndim == 3:
        print(f"Flattening 3D output projection {model['fast_output.weight'].shape}")
        model["fast_output.weight"] = (
            model["fast_output.weight"].transpose(0, 1).flatten(1)
        )
    save_file(model, "model.safetensors")


if __name__ == "__main__":
    main()
