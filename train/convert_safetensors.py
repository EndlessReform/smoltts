import sys
from safetensors.torch import save_file
import torch


def main():
    data = torch.load(sys.argv[1])
    model = data["model_state_dict"]
    model = {key.replace("_orig_mod.", ""): value for key, value in model.items()}
    if model["fast_output.weight"].ndim == 3:
        print(f"Flattening 3D output projection {model['fast_output.weight'].shape}")
        w = model["fast_output.weight"]  # [codebooks, hidden_dim, codebook_size]
        model["fast_output.weight"] = model["fast_output.weight"] = (
            w.permute(1, 0, 2).reshape(768, -1).T.contiguous()
        )
    save_file(model, "model.safetensors")


if __name__ == "__main__":
    main()
