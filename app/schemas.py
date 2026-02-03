"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional

class DetectionRequest(BaseModel):
    """Request schema for voice detection"""
    audioData: str = Field(..., description="Base64 encoded audio data")
    language: str = Field(..., description="Language of the audio (english, tamil, hindi, malayalam, telugu)")
    userId: str = Field(..., description="User identifier")

class DetectionResponse(BaseModel):
    """Response schema for voice detection"""
    status: str = Field(..., description="Status of the request")
    language: str = Field(..., description="Detected language")
    classification: str = Field(..., description="HUMAN or AI_GENERATED")
    confidenceScore: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    explanation: str = Field(..., description="Explanation of the classification")