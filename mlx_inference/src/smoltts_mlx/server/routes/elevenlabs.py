from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal

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
    content, media_type = core.generate_audio(
        input_text=item.text,
        voice=voice_id,
        response_format=output_format,
    )

    return Response(
        content=content,
        media_type=media_type,  # or audio/l16 for 16-bit PCM
        headers={
            "Content-Disposition": f'attachment; filename="elevenlabs_speech.{output_format.split("_")[0]}"',
            "X-Sample-Rate": output_format.split("_")[1]
            if output_format is not None
            else "24000",
        },
    )


@router.post("/text-to-speech/{voice_id}/stream")
async def stream_tts(
    voice_id: str,
    item: CreateSpeechRequest,
    http_request: Request,
    output_format: Literal["pcm_24000"] = "pcm_24000",
):
    core = http_request.app.state.tts_core

    def generate():
        for audio_chunk in core.stream_audio(item.text, voice=voice_id):
            yield audio_chunk

    return StreamingResponse(
        generate(),
        media_type="audio/mpeg" if output_format.startswith("mp3_") else "audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="speech.{output_format.split("_")[0]}"',
            # "X-Sample-Rate": output_format.split("_")[1],
            "X-Sample-Rate": "24000",
        },
    )
