from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/v1", tags=["ElevenLabs"])


class CreateSpeechRequest(BaseModel):
    text: str
    model_id: Optional[str]


@router.post("/text-to-speech/{voice_id}")
async def text_to_speech_blocking(
    voice_id: str, item: CreateSpeechRequest, http_request: Request
):
    core = http_request.app.state.tts_core
    wav_data = core.generate_audio(
        input_text=item.text,
        voice=voice_id,  # or map to your internal speaker IDs
        response_format="wav",
    )
    return Response(
        content=wav_data,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="elevenlabs_speech.wav"'},
    )
