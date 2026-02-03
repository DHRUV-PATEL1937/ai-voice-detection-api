"""
FastAPI Application - AI Voice Detection API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
from pathlib import Path

# Import detector
from app.detector import VoiceDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Detect AI-generated voices in multiple languages",
    version="1.0.0"
)

# CRITICAL: Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Request/Response models
class VoiceDetectionRequest(BaseModel):
    audioData: str  # Base64 encoded audio
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
    print("="*60 + "\n")
    
    # Download model if on Render
    import os
    if os.getenv('RENDER'):
        print("📦 Downloading model on Render...")
        try:
            from scripts.download_model import download_and_extract_model
            download_and_extract_model()
            print("✅ Model downloaded")
        except Exception as e:
            print(f"⚠️ Model download warning: {e}")
    
    # Initialize detector
    try:
        detector = VoiceDetector()
        print("✅ Detector initialized")
    except Exception as e:
        print(f"⚠️ Detector initialization warning: {e}")
        print("⚠️ API will start but detection may fail")
    
    print("\n" + "="*60)
    print("✅ API Ready!")
    print("="*60 + "\n")

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_ready": detector is not None
    }

@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def detect_voice(request: VoiceDetectionRequest):
    """Detect if voice is AI-generated or human"""
    
    logger.info(f"📥 Detection request - Language: {request.language}, User: {request.userId}")
    
    if detector is None:
        logger.error("❌ Detector not initialized")
        raise HTTPException(
            status_code=503,
            detail="Model is still loading, please try again in a moment"
        )
    
    try:
        # Perform detection
        result = detector.detect(
            base64_audio=request.audioData,
            language=request.language
        )
        
        logger.info(f"✅ Detection complete - {result['classification']} ({result['confidenceScore']:.2f})")
        
        return VoiceDetectionResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Detection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

# Mount static files AFTER defining routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)