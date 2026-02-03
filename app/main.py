"""
FastAPI Application - AI Voice Detection API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import logging
from pathlib import Path
import os

# Import detector
# Ensure app.detector exists or mock it if necessary for build
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
    description="Detect AI-generated voices in multiple languages",
    version="1.0.0"
)

# CRITICAL: Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class VoiceDetectionRequest(BaseModel):
    audioData: str
    language: str
    userId: str

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
    """Initialize detector on startup"""
    global detector
    print("\n" + "="*60)
    print("🚀 Starting AI Voice Detection API...")
    
    # Download model if on Render
    if os.getenv('RENDER'):
        print("📦 Downloading model on Render...")
        try:
            from scripts.download_model import download_and_extract_model
            download_and_extract_model()
            print("✅ Model downloaded/verified")
        except Exception as e:
            print(f"⚠️ Model download warning: {e}")
    
    # Initialize detector
    try:
        if VoiceDetector:
            detector = VoiceDetector()
            print("✅ Detector initialized")
        else:
            print("⚠️ VoiceDetector class not imported")
    except Exception as e:
        print(f"⚠️ Detector initialization warning: {e}")
    
    print("✅ API Ready!")
    print("="*60 + "\n")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "detector_ready": detector is not None}

@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def detect_voice(request: VoiceDetectionRequest):
    logger.info(f"📥 Detection request - Language: {request.language}")
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is loading")
    
    try:
        result = detector.detect(base64_audio=request.audioData, language=request.language)
        return VoiceDetectionResponse(**result)
    except Exception as e:
        logger.error(f"❌ Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# STATIC FILE CONFIGURATION (The Fix)
# ---------------------------------------------------------

# 1. Mount static files FIRST to /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Serve index.html at root
@app.get("/")
async def root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)