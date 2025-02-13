import io
import mlx.core as mx
import numpy as np
from pydub import AudioSegment
from scipy import signal
import soundfile as sf
import time
from typing import Union
from tqdm import tqdm

from smoltts_mlx.io.wav import pcm_to_wav_bytes
from smoltts_mlx.lm.cache import make_prompt_cache
from smoltts_mlx.lm.generate import generate_blocking, SingleBatchGenerator


class TTSCore:
    def __init__(self, model, tokenizer, mimi_tokenizer, prompt_encoder, settings):
        self.model = model
        self.tokenizer = tokenizer
        self.mimi_tokenizer = mimi_tokenizer
        self.prompt_encoder = prompt_encoder
        self.settings = settings

    def resolve_speaker_id(self, voice: Union[str, int]) -> int:
        # TODO: Fix speaker cache
        if isinstance(voice, int):
            return voice
        elif isinstance(voice, str) and voice.isnumeric():
            return int(voice)
        return 0

    def get_prompt(self, input_text: str, voice: Union[str, int]) -> mx.array:
        speaker_id = self.resolve_speaker_id(voice)
        sysprompt = self.prompt_encoder.encode_text_turn(
            "system", f"<|speaker:{speaker_id}|>"
        )
        user_prompt = self.prompt_encoder.encode_text_turn("user", input_text)
        assistant_prefix = self.prompt_encoder.encode_text_turn("assistant")
        prompt = mx.concat([sysprompt, user_prompt, assistant_prefix], axis=1)[
            mx.newaxis, :, :
        ]
        return prompt

    def generate_audio(
        self, input_text: str, voice: Union[str, int], response_format: str = "wav"
    ):
        prompt = self.get_prompt(input_text, voice)
        # Generate semantic tokens
        gen = generate_blocking(
            self.model, prompt, self.settings.generation, audio_only=True
        )

        # Convert to numpy and decode
        # tokens = np.array(gen).astype(np.uint32)
        start_time = time.time()
        out = self.mimi_tokenizer.decode(gen)
        mx.eval(out)
        end_time = time.time()
        print(f"Took {end_time - start_time:.2f}s to decode")
        # print(f"Took {end_time - start_time:.2f}s to decode")

        start_time = time.time()
        pcm_data = np.array(out)
        audio_data, media_type = self.format_audio_chunk(
            pcm_data.flatten(), response_format
        )
        end_time = time.time()
        print(f"Took {end_time - start_time:.2f}s to transcode")
        mx.metal.clear_cache()

        return audio_data, media_type

    def stream_audio(self, input_text: str, voice: Union[str, int]):
        prompt = self.get_prompt(input_text, voice)
        token_gen = SingleBatchGenerator(
            self.model, prompt, self.settings.generation, audio_only=True
        )
        mimi_cache = make_prompt_cache(self.mimi_tokenizer.decoder_transformer)

        # TODO REMOVE THIS this is just for gut-check
        # all_pcm = []

        for token in tqdm(token_gen):
            audio_tokens = token.vq_tensor[:, 1:, :]
            # print(f"Shape: {np_tokens.shape}")
            pcm_chunk = self.mimi_tokenizer.decode_step(audio_tokens, mimi_cache)
            if pcm_chunk is not None:
                # all_pcm.append(pcm_chunk)
                audio_data = np.array(pcm_chunk).flatten().tobytes()
                yield audio_data
                # audio_data = self.format_audio_chunk(pcm_chunk, output_format)

                # yield audio_data
        self.mimi_tokenizer.decoder.reset()
        # pcm_chunk = np.array(mx.concat(all_pcm, axis=-1).flatten())
        mx.metal.clear_cache()
        # yield self.format_audio_chunk(pcm_chunk, output_format)

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
