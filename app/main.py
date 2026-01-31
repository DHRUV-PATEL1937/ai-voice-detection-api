"""
FastAPI Application for AI Voice Detection
"""
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import DetectionRequest, DetectionResponse, ErrorResponse
from app.detector import VoiceDetector
from app.auth import verify_api_key, get_api_key_info
from config.settings import API_TITLE, API_VERSION, API_DESCRIPTION, SUPPORTED_LANGUAGES

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup"""
    global detector
    print("\n" + "="*60)
    print("🚀 STARTING AI VOICE DETECTION API")
    print("="*60)
    
    try:
        detector = VoiceDetector()
        print("✅ Detector initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        raise
    
    print("="*60)
    print(f"📡 API ready at http://localhost:8000")
    print(f"📚 Documentation at http://localhost:8000/docs")
    print("="*60 + "\n")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "AI Voice Detection API",
        "version": API_VERSION,
        "status": "online",
        "supported_languages": SUPPORTED_LANGUAGES,
        "endpoints": {
            "detection": "/api/voice-detection",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": detector is not None,
        "supported_languages": SUPPORTED_LANGUAGES
    }


@app.post(
    "/api/voice-detection",
    response_model=DetectionResponse,
    responses={
        200: {"model": DetectionResponse},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Detection"]
)
async def detect_voice(
    request: DetectionRequest,
    x_api_key: str = Header(..., alias="x-api-key", description="Your API key")
):
    """
    Detect if audio is AI-generated or Human
    
    - **language**: One of Tamil, English, Hindi, Malayalam, Telugu
    - **audioFormat**: Must be 'mp3'
    - **audioBase64**: Base64 encoded MP3 audio file
    
    Returns classification with confidence score and explanation.
    """
    
    start_time = time.time()
    
    # Validate API key
    if not verify_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    # Get API key info for logging
    key_info = get_api_key_info(x_api_key)
    print(f"📞 Request from: {key_info.get('user', 'unknown')} ({request.language})")
    
    # Validate language
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Language must be one of: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    
    # Validate audio format
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only MP3 format is supported"
        )
    
    try:
        # Perform detection
        result = detector.detect(
            base64_audio=request.audioBase64,
            language=request.language
        )
        
        processing_time = time.time() - start_time
        print(f"   ✅ Classified as {result['classification']} ({result['confidenceScore']:.2f}) in {processing_time:.2f}s")
        
        return DetectionResponse(**result)
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )