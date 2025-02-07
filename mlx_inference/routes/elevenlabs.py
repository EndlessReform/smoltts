from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, Field
from typing import Optional
from scipy import signal
import soundfile as sf
import io

router = APIRouter(prefix="/v1", tags=["ElevenLabs"])


class CreateSpeechRequest(BaseModel):
    text: str
    model_id: Optional[str] = Field(default=None)


@router.post("/text-to-speech/{voice_id}")
async def text_to_speech_blocking(
    voice_id: str,
    item: CreateSpeechRequest,
    http_request: Request,
    output_format: Optional[str] = Query(
        None, description="Desired output format. No MP3 support"
    ),
):
    core = http_request.app.state.tts_core
    float_pcm = core.generate_audio(
        input_text=item.text,
        voice=voice_id,  # or map to your internal speaker IDs
        response_format="wav",
    ).flatten()

    output_format = output_format if output_format is not None else "pcm_24000"
    sample_rate = int(output_format.split("_")[1])
    if sample_rate != 24000:
        num_samples = int(len(float_pcm) * sample_rate / 24000)
        float_pcm = signal.resample(float_pcm, num_samples)
    mem_buf = io.BytesIO()
    sf.write(mem_buf, float_pcm, sample_rate, format="raw", subtype="PCM_16")

    if output_format.startswith("pcm_"):
        # Extract sample rate from format string
        # Resample if needed
        content = bytes(mem_buf.getbuffer())
        media_type = "audio/x-pcm"

    elif output_format.startswith("mp3_"):
        from pydub import AudioSegment

        # Now create AudioSegment from the properly converted PCM
        audio = AudioSegment(
            data=bytes(mem_buf.getbuffer()),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )

        # Export to MP3
        out_buf = io.BytesIO()
        audio.export(out_buf, format="mp3", bitrate=output_format.split("_")[-1] + "k")
        content = out_buf.getvalue()
        media_type = "audio/mpeg"

    else:
        # TODO
        raise NotImplementedError(f"Format {output_format} not yet supported")

    return Response(
        content=content,
        media_type=media_type,  # or audio/l16 for 16-bit PCM
        headers={
            "Content-Disposition": f'attachment; filename="elevenlabs_speech.{output_format.split("_")[0]}"',
            "X-Sample-Rate": str(sample_rate),
        },
    )
