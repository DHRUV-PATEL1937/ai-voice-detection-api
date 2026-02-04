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
    allow_origins=["*"],  # Allows all origins (Firebase, Localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers (API Key, Content-Type)
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
    print("🚀 Starting AI Voice Detection API (Backend)...")
    print("="*60 + "\n")
    
    # Download model if on Render
    if os.getenv('RENDER'):
        print("📦 Checking environment...")
        try:
            from scripts.download_model import download_and_extract_model
            download_and_extract_model()
        except ImportError:
            pass 
        except Exception as e:
            print(f"⚠️ Model download warning: {e}")
    
    # Initialize detector
    try:
        if VoiceDetector:
            detector = VoiceDetector()
            print("✅ Detector initialized successfully")
        else:
            print("❌ VoiceDetector class could not be imported")
    except Exception as e:
        print(f"❌ Detector initialization failed: {e}")
        print("⚠️ API will start, but detection endpoints will error.")
    
    print("\n" + "="*60)
    print("✅ API Ready to accept connections!")
    print("="*60 + "\n")

@app.get("/")
async def root():
    """Root endpoint - JSON info only (No UI)"""
    return {
        "status": "online",
        "message": "AI Voice Detection API is running. Connect via Frontend.",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_ready": detector is not None
    }

# ✅ NEW: Handle GET requests to prevent 405 errors and "loops"
@app.get("/api/voice-detection")
async def get_voice_detection_info():
    return {
        "status": "online",
        "message": "This endpoint expects a POST request with audio data. Use the frontend to upload a file."
    }

@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def detect_voice(request: VoiceDetectionRequest):
    """Detect if voice is AI-generated or human"""
    
    logger.info(f"📥 Request received - Language: {request.language}, User: {request.userId}")
    
    if detector is None:
        logger.error("❌ Detector not initialized")
        raise HTTPException(
            status_code=503,
            detail="Model is still loading or failed to initialize. Please check server logs."
        )
    
    try:
        # Perform detection
        result = detector.detect(
            base64_audio=request.audioData,
            language=request.language
        )
        
        logger.info(f"✅ Analysis complete: {result['classification']} ({result['confidenceScore']:.2f})")
        return VoiceDetectionResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)