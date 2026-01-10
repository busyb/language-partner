from pydantic import BaseModel


class GenerateSentenceRequest(BaseModel):
    language: str  # "en", "es", "fr", "zh"

class SynthesizeRequest(BaseModel):
    text: str
    language: str