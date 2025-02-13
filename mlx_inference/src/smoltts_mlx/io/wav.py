import numpy as np


def pcm_to_wav_bytes(pcm_data: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert raw PCM data to WAV bytes.

    Args:
        pcm_data: Floating point PCM data in range [-1, 1], will be flattened
        sample_rate: Sample rate in Hz (default: 24000)

    Returns:
        Complete WAV file as bytes
    """
    # Flatten to 1D array first
    pcm_data = pcm_data.flatten()

    header = bytearray()
    # RIFF chunk
    header.extend(b"RIFF")
    header.extend((len(pcm_data) * 2 + 36).to_bytes(4, "little"))  # Total size minus 8
    header.extend(b"WAVE")
    # fmt chunk
    header.extend(b"fmt ")
    header.extend((16).to_bytes(4, "little"))  # fmt chunk size
    header.extend((1).to_bytes(2, "little"))  # PCM format
    header.extend((1).to_bytes(2, "little"))  # Mono
    header.extend(sample_rate.to_bytes(4, "little"))
    header.extend((sample_rate * 2).to_bytes(4, "little"))  # Bytes per second
    header.extend((2).to_bytes(2, "little"))  # Block align
    header.extend((16).to_bytes(2, "little"))  # Bits per sample
    # data chunk
    header.extend(b"data")
    header.extend((len(pcm_data) * 2).to_bytes(4, "little"))  # Data size

    # Convert PCM to 16-bit samples
    wav_data = (pcm_data * 32767).astype(np.int16).tobytes()
    return bytes(header) + wav_data
