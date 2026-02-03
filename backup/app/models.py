"""
Pydantic Models for API Request/Response Validation
"""
from pydantic import BaseModel, Field, validator
from typing import Literal
import base64

class DetectionRequest(BaseModel):
    """Request model for voice detection"""
    
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ...,
        description="Language of the audio file"
    )
    
    audioFormat: str = Field(
        ...,
        pattern="^mp3$",
        description="Audio format (must be mp3)"
    )
    
    audioBase64: str = Field(
        ...,
        min_length=100,
        description="Base64 encoded MP3 audio"
    )
    
    @validator('audioBase64')
    def validate_base64(cls, v):
        """Validate base64 encoding"""
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError('Invalid base64 encoding')
    
    class Config:
        schema_extra = {
            "example": {
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
            }
        }


class DetectionResponse(BaseModel):
    """Response model for voice detection"""
    
    status: Literal["success", "error"]
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "language": "Tamil",
                "classification": "AI_GENERATED",
                "confidenceScore": 0.91,
                "explanation": "Detected unnaturally consistent pitch and overly stable energy levels characteristic of synthetic speech"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    
    status: Literal["error"]
    message: str
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        }