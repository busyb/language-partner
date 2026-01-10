from pydantic import BaseModel


class TranslateRequest(BaseModel):
    text: str
    from_code: str  # e.g. "en"
    to_code: str  # e.g. "es"


class TranslateResponse(BaseModel):
    text: str
    text_language: str
    latency_ms: float

