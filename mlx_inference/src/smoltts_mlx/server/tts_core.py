import io
import mlx.core as mx
import numpy as np
from pydub import AudioSegment
from scipy import signal
import soundfile as sf
import time
from typing import Union
from tqdm import tqdm

from smoltts_mlx import SmolTTS
from smoltts_mlx.io.wav import pcm_to_wav_bytes


class TTSCore:
    def __init__(self, model: SmolTTS, settings):
        self.model = model
        self.settings = settings

    def resolve_speaker_id(self, voice: Union[str, int]) -> int:
        # TODO: Fix speaker cache
        if isinstance(voice, int):
            return voice
        elif isinstance(voice, str) and voice.isnumeric():
            return int(voice)
        return 0

    def generate_audio(
        self, input_text: str, voice: Union[str, int], response_format: str = "wav"
    ):
        start_time = time.time()
        pcm_data = self.model(input_text, str(voice))
        end_time = time.time()
        print(f"Took {end_time - start_time:.2f}s to decode")

        start_time = time.time()
        audio_data, media_type = self.format_audio_chunk(
            pcm_data.flatten(), response_format
        )
        end_time = time.time()
        print(f"Took {end_time - start_time:.2f}s to transcode")
        mx.metal.clear_cache()

        return audio_data, media_type

    def stream_audio(self, input_text: str, voice: Union[str, int]):
        for pcm_chunk in tqdm(self.model.stream(input_text, str(voice))):
            if pcm_chunk is not None:
                audio_data = pcm_chunk.tobytes()
                yield audio_data

    def format_audio_chunk(
        self, pcm_data: np.ndarray, output_format: str = "pcm_24000"
    ) -> tuple[bytes, str]:
        """Format a chunk of PCM data into the requested format.
        Returns (formatted_bytes, media_type)"""
        sample_rate = int(output_format.split("_")[1])
        pcm_data = pcm_data.flatten()

        # Resample if needed
        if sample_rate != 24000:
            num_samples = int(len(pcm_data) * sample_rate / 24000)
            pcm_data = signal.resample(pcm_data, num_samples)

        # Convert to 16-bit PCM first
        mem_buf = io.BytesIO()
        sf.write(mem_buf, pcm_data, sample_rate, format="raw", subtype="PCM_16")
        pcm_bytes = bytes(mem_buf.getbuffer())

        if output_format.startswith("pcm_"):
            return pcm_bytes, "audio/x-pcm"
        elif output_format.startswith("wav_"):
            wav_bytes = pcm_to_wav_bytes(pcm_data=pcm_data, sample_rate=sample_rate)
            return wav_bytes, "audio/wav"
        elif output_format.startswith("mp3_"):
            bitrate = output_format.split("_")[-1]
            audio = AudioSegment(
                data=pcm_bytes,
                sample_width=2,
                frame_rate=sample_rate,
                channels=1,
            )
            out_buf = io.BytesIO()
            audio.export(out_buf, format="mp3", bitrate=f"{bitrate}k")
            return out_buf.getvalue(), "audio/mpeg"
        else:
            raise NotImplementedError(f"Format {output_format} not yet supported")
