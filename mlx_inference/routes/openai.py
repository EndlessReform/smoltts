from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field
from typing import Literal, Union


class SpeechRequest(BaseModel):
    model: str = Field(default="tts-1-hd")
    input: str
    voice: Union[str, int] = Field(default="alloy")
    response_format: Literal["wav"] = Field(default="wav")


router = APIRouter(prefix="/v1", tags=["OpenAI"])


@router.post("/audio/speech")
async def openai_speech(item: SpeechRequest, http_request: Request):
    core = http_request.app.state.tts_core
    wav_data = core.generate_audio(
        input_text=item.input, voice=item.voice, response_format=item.response_format
    )
    return Response(
        wav_data,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )
