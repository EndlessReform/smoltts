import math
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import MimiModel
from typing import List

SAMPLING_RATE = 24_000
CODEC_HZ = 12.5


def get_target_length(
    arr: torch.Tensor, sampling_rate=SAMPLING_RATE, codec_hz=CODEC_HZ
) -> int:
    return math.ceil(arr.size(-1) / (sampling_rate / codec_hz))


class MimiCodec:
    def __init__(self, model_name: str = "kyutai/mimi", device: str = "cuda"):
        model = MimiModel.from_pretrained(model_name)
        model = model.to(device)
        self.model = model
        self.device = device

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim == 2:
            # Add spurious batch dimension
            codes = codes.unsqueeze(0)

        with torch.no_grad():
            out_pcm = self.model.decode(codes.to(self.device))
        out_tensor = out_pcm.audio_values[0].detach().to("cpu")
        # Trim final frame to prevent random artifacts
        return out_tensor[:, :-1]

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            # Single mono audio
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.ndim == 2:
            # Dual channel
            audio = audio.unsqueeze(0)
        else:
            raise ValueError(
                f"Use batch endpoint to encode audio safely; got {audio.ndim} dims but expected channel, seqlen or seqlen"
            )

        with torch.no_grad():
            encoded = self.model.encode(audio.to(self.device))
        codes = encoded.audio_codes[:, 0:8, :].clone().cpu()
        del encoded
        torch.cuda.empty_cache()
        return codes.squeeze(0)

    def encode_batch(self, audios: List[torch.Tensor]) -> List[torch.Tensor]:
        target_lengths = [get_target_length(arr) - 1 for arr in audios]
        # Add spurious channel dimension, audio should be mono
        padded_batch = pad_sequence(audios, batch_first=True).unsqueeze(1)
        padding_mask = (padded_batch != 0).float()

        with torch.no_grad():
            encoder_outputs = self.model.encode(
                padded_batch.to(self.device), padding_mask=padding_mask.to(self.device)
            )
        codes = encoder_outputs.audio_codes[:, 0:8, :].cpu()
        del padded_batch
        del encoder_outputs
        torch.cuda.empty_cache()

        chunked = list(torch.unbind(codes, dim=0))
        return [t[:, :length] for t, length in zip(chunked, target_lengths)]
