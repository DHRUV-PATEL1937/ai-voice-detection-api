"""
AI Voice Detection API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.detector import VoiceDetector
from app.schemas import DetectionRequest, DetectionResponse

# Initialize FastAPI app
app = FastAPI(title="AI Voice Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = VoiceDetector()

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def root():
    """Serve the main page"""
    index_file = Path(__file__).parent.parent / "static" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "healthy", "message": "AI Voice Detection API is running"}

@app.post("/api/voice-detection", response_model=DetectionResponse)
async def detect_voice(request: DetectionRequest):
    """Detect if voice is AI-generated or human"""
    
    try:
        print(f"📞 Request from: {request.userId} ({request.language})")
        
        result = detector.detect(
            base64_audio=request.audioData,
            language=request.language
        )
        
        print(f"   ✅ Classified as {result['classification']} ({result['confidenceScore']*100:.2f}%)")
        
        return DetectionResponse(**result)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))