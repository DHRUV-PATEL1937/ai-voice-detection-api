"""
FastAPI Application - AI Voice Detection API (Backend Only)
"""
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from pathlib import Path
import os
import sys

# Setup Python path to find modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import detector
try:
    from app.detector import VoiceDetector
except ImportError:
    VoiceDetector = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Backend API for detecting AI-generated voices",
    version="1.0.0"
)

# CRITICAL: CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ UPDATED: Request model to match Hackathon Tester
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str       # Changed to match Tester
    audioBase64: str       # Changed from audioData to audioBase64
    # userId removed as Tester doesn't send it

class VoiceDetectionResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

# Initialize detector
detector = None

@app.on_event("startup")
async def startup_event():
    global detector
    print("🚀 Starting API...")
    
    if os.getenv('RENDER'):
        try:
            from scripts.download_model import download_and_extract_model
            download_and_extract_model()
        except: pass
    
    try:
        if VoiceDetector:
            detector = VoiceDetector()
            print("✅ Detector initialized")
    except Exception as e:
        print(f"❌ Detector failed: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "detector_ready": detector is not None}

# GET Handler to prevent 405 errors
@app.get("/api/voice-detection")
async def get_voice_detection_info():
    return {
        "status": "online",
        "message": "Send POST request with audioBase64"
    }

@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
def detect_voice(request: VoiceDetectionRequest):
    """Detect if voice is AI-generated or human"""
    
    # Updated logging to match new fields
    logger.info(f"📥 Request received - Language: {request.language}, Format: {request.audioFormat}")
    
    if detector is None:
        logger.error("❌ Detector not initialized")
        raise HTTPException(status_code=503, detail="Model is loading")
    
    try:
        # Perform detection using audioBase64
        result = detector.detect(
            base64_audio=request.audioBase64,  # ✅ Using new field name
            language=request.language
        )
        
        logger.info(f"✅ Analysis complete: {result['classification']} ({result['confidenceScore']:.2f})")
        return VoiceDetectionResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)