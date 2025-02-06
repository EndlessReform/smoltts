# Pretraining

## Config format

Create your own run config in the `../configs/yourmodel` folder.
The options should be pretty self-explanatory.

## Starting a run

Training is currently only tested with CUDA and Linux. For example:

```bash
uv run main.py --config ../config/kokoro_v1/scaleup.json
```

Artifacts will be saved to the `../checkpoints/` folder under a run ID.

## Extracting `model.safetensors` from checkpoint

```bash
# Replace with whatever run you want
uv run convert_safetensors.py ../config/your-run-id-here/step_somestep.pt
```

will save `model.safetensors` to this folder.

This will be improved.
