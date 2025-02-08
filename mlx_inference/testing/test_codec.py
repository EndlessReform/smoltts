from huggingface_hub import hf_hub_download
import mlx.core as mx

from mlx_inference.codec.mimi import MimiConfig, MimiModel
from mlx_inference.codec.conv import SeanetConfig
from mlx_inference.codec.rvq import RVQConfig
from mlx_inference.codec.transformer import MimiTransformerConfig


def main():
    config = MimiConfig(
        seanet=SeanetConfig(), transformer=MimiTransformerConfig(), rvq=RVQConfig()
    )
    model_path = hf_hub_download("kyutai/mimi", "model.safetensors")
    print("Downloaded file")

    model = MimiModel(config)
    print("Setup")
    state_dict = mx.load(model_path)

    # Yes, this is dumb.
    # The all-knowing maintainers of MLX decided to serialize conv1ds as NHWC instaed of NCHW,
    # despite the entire API surface being designed to mimic pytorch, because it's faster on apple silicon, and then
    # "helpfully" leaked that abstraction onto me.
    # But the convtrans1d is DIFFERENT yet again.
    # All hail Apple.
    def is_convtrans1d(key) -> bool:
        return (
            # Decoder only
            "decoder" in key
            and key.endswith(".conv.weight")
            and "block" not in key
            # Layer 0 is regular
            and "0" not in key
            # Final layer is regular
            and "14" not in key
        )

    def is_conv1d(key):
        return (
            key.endswith(".conv.weight")
            # Patched
            and key != "upsample.conv.weight"
            # RVQ proj
            or "quantizer.input_proj" in key
            or "quantizer.output_proj" in key
        )

    converted_dict = {
        k: v.transpose(1, 2, 0)
        if is_convtrans1d(k)
        else v.transpose(0, 2, 1)
        if is_conv1d(k)
        else v
        for k, v in state_dict.items()
    }
    # print(converted_dict.keys())
    # print(
    #     "Dict structure:",
    #     {k: converted_dict[k].shape for k in list(converted_dict.keys())[:5]},
    # )
    # print("Tree structure:", list(converted_dict.keys())[0].split("."))
    weight_list = [(k, v) for k, v in converted_dict.items()]

    model.load_weights(weight_list, strict=True)
    mx.eval(model.parameters())
    model.eval()
    print("Model loaded")


if __name__ == "__main__":
    main()
