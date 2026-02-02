"""
FastAPI Application for AI Voice Detection
"""

import sys
from pathlib import Path
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# -------------------------------------------------
# Add project root to Python path
# -------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

# -------------------------------------------------
# ✅ AUTO DOWNLOAD MODEL IF NOT PRESENT
# -------------------------------------------------
from scripts.download_model import download_model_from_gdrive

try:
    print("🔍 Checking for model...")
    download_model_from_gdrive()
except Exception as e:
    print(f"⚠️  Model download warning: {e}")
    print("   Please ensure model is uploaded or MODEL_FILE_ID is set")

# -------------------------------------------------
# App Imports
# -------------------------------------------------
from app.models import DetectionRequest, DetectionResponse, ErrorResponse
from app.detector import VoiceDetector
from app.auth import verify_api_key, get_api_key_info
from config.settings import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    SUPPORTED_LANGUAGES
)

# -------------------------------------------------
# Create FastAPI App
# -------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# -------------------------------------------------
# ✅ UPDATED CORS CONFIGURATION
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://your-hackathon-website.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Static Files & Templates
# -------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------------------------------------
# Global Detector Instance
# -------------------------------------------------
detector = None


# -------------------------------------------------
# Startup Event
# -------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global detector
    print("\n" + "=" * 60)
    print("🚀 STARTING AI VOICE DETECTION API")
    print("=" * 60)

    try:
        detector = VoiceDetector()
        print("✅ Detector initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        raise

    print("=" * 60)
    print("📡 API running at http://localhost:8000")
    print("🎨 Web UI at http://localhost:8000")
    print("📚 Docs at http://localhost:8000/docs")
    print("=" * 60 + "\n")


# -------------------------------------------------
# UI Route
# -------------------------------------------------
@app.get("/", tags=["UI"], include_in_schema=False)
async def ui_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -------------------------------------------------
# API Info
# -------------------------------------------------
@app.get("/api/info", tags=["Info"])
async def api_info():
    return {
        "message": "AI Voice Detection API",
        "version": API_VERSION,
        "status": "online",
        "supported_languages": SUPPORTED_LANGUAGES,
        "endpoints": {
            "ui": "/",
            "detection": "/api/voice-detection",
            "health": "/health",
            "docs": "/docs"
        }
    }


# -------------------------------------------------
# Health Check
# -------------------------------------------------
@app.get("/health", tags=["Info"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": detector is not None,
        "supported_languages": SUPPORTED_LANGUAGES
    }


# -------------------------------------------------
# Voice Detection Endpoint
# -------------------------------------------------
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
    x_api_key: str = Header(..., alias="x-api-key")
):
    """
    Detect if audio is AI-generated or Human
    """

    start_time = time.time()

    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    key_info = get_api_key_info(x_api_key)
    print(f"📞 Request from: {key_info.get('user', 'unknown')} ({request.language})")

    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Language must be one of: {', '.join(SUPPORTED_LANGUAGES)}"
        )

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only MP3 format is supported"
        )

    try:
        result = detector.detect(
            base64_audio=request.audioBase64,
            language=request.language
        )

        processing_time = time.time() - start_time
        print(
            f"   ✅ Classified as {result['classification']} "
            f"({result['confidenceScore']:.2f}) in {processing_time:.2f}s"
        )

        return DetectionResponse(**result)

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


# -------------------------------------------------
# Global Exception Handler
# -------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error occurred"
        }
    )


# -------------------------------------------------
# Run Server
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
