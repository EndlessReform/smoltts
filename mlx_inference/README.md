# smolltts-mlx

## Installation

Requires working Python instance and Apple Silicon Mac.

```bash
pip install smoltts-mlx
```

Or if you have [`uv`](https://docs.astral.sh/uv/) (hint hint), simply use [uvx](https://docs.astral.sh/uv/guides/tools/):

```bash
ux --from smoltts_mlx smoltts-server
```

## Server

### Startup

From the CLI, run:

```bash
smoltts-server
```

Options:

- `--port` (optional): Port to listen on (default: 8000)
- `--config` (optional): Point to a JSON file. (See below for spec)

### Supported voices

As of February 2025, we support these voices from Kokoro:

- **American:** heart (default), bella, nova, sky, sarah, michael, fenrir, liam
- **British:** emma, isabella, fable

Voice cloning is currently not supported, but coming soon!

Unfortunately, GitHub doesn't support audio previews, but check out `docs/examples` for samples.

### ElevenLabs endpoints

We support the following two ElevenLabs endpoints (more to come):

- [`/v1/text-to-speech/$ID`](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) with MP3, WAV, and PCM output
- [`/v1/text-to-speech/$ID/stream`](https://elevenlabs.io/docs/api-reference/text-to-speech/convert-as-stream) (PCM-only for now)

Here's an example with the Python SDK:

```python
from elevenlabs.client import ElevenLabs

client = ElevenLabs(
    # or wherever you're running this on
    base_url="http://localhost:8000",
)

request_gen = client.text_to_speech.convert(
    voice_id="0",
    output_format="mp3_44100_128",
    text="You can turn on latency optimizations at some cost of quality. The best possible final latency varies by model.",
)
```

### OpenAI endpoints

We support `/v1/audio/speech` (MP3 and WAV).

Here's an example with the [OpenAI Python SDK](https://platform.openai.com/docs/guides/text-to-speech#quickstart):

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000"
)

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Today is a wonderful day to build something people love!",
)
response.stream_to_file(speech_file_path)
```

### Configuration

Default settings are stored by default at `~/Library/Cache/smoltts`.

You can also specify a JSON file with `--config`.

```json
{
  // "checkpoint_dir": "../inits/foobar/"
  "model_id": "jkeisling/smoltts_v0",
  "generation": {
    "default_temp": 0.0,
    "default_fast_temp": 0.5,
    "min_p": 0.1
  },
  "model_type": {
    "family": "dual_ar",
    "codec": "mimi",
    "version": null
  }
}
```

## Library

### Basic Usage

```python
from smoltts_mlx import SmolTTS
from IPython.display import Audio

# Initialize model (downloads weights automatically)
model = SmolTTS()

# Basic generation to numpy PCM array
pcm = model("Hello world!")
Audio(pcm, rate=model.sampling_rate)

# Streaming generation for real-time audio
for pcm_chunk in model.stream("This is a longer piece of text to stream."):
    # Yields 80ms PCM frames as they're generated
    process_audio(pcm_chunk)
```

### Voice Selection

```python
# Use a specific voice
pcm = model("Hello!", voice="af_bella")

# Create a custom voice from reference audio
speaker_prompt = model.create_speaker(
    system_prompt="<|speaker:0|>",
    samples=[{
        "text": "This is a sample sentence.",
        "audio": reference_audio  # Numpy array of PCM data
    }]
)

# Generate with custom voice
pcm = model(
    "Using a custom voice created from reference audio.",
    speaker_prompt=speaker_prompt
)
```

### Working with Audio

The model works with raw PCM audio at 24kHz sample rate. For format conversion:

```python
import soundfile as sf

# Save to WAV
sf.write("output.wav", pcm, model.sampling_rate)

# Load reference audio
audio, sr = sf.read("reference.wav")
if sr != model.sampling_rate:
    # Resample if needed
    audio = resampy.resample(audio, sr, model.sampling_rate)
```

### Advanced Configuration

```python
# Use custom model weights
model = SmolTTS(
    model_id="path/to/custom/model",
    checkpoint_dir="/path/to/local/weights"
)

# Access underlying components
mimi_codes = model.codec.encode(pcm)  # Work with Mimi tokens directly
```

### Performance Notes

- First generation may be slower due to model loading and warmup
- Use streaming for longer texts to begin playback before full generation

### Requirements

- Apple Silicon Mac (M1/M2/M3)
- Python 3.9 or later

## Developing locally

Please use [uv](https://docs.astral.sh/uv/).

From the root of this repo:

```bash
uv sync --all-packages
```

### Creating CSM init
