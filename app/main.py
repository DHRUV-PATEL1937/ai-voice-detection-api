"""
AI Voice Detection API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Global variables
detector = None
model_loaded = False

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global detector, model_loaded
    
    print("🚀 Starting AI Voice Detection API...")
    
    # Download model if on Render
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
        from app.detector import VoiceDetector
        detector = VoiceDetector()
        model_loaded = True
        print("✅ Detector initialized")
    except Exception as e:
        print(f"⚠️ Detector initialization warning: {e}")
        print("⚠️ API will start but detection may fail")

from app.schemas import DetectionRequest, DetectionResponse

@app.get("/")
async def root():
    """Serve the main page"""
    try:
        # Try to find index.html
        possible_paths = [
            Path(__file__).parent.parent / "static" / "index.html",
            Path("static") / "index.html",
            Path("/opt/render/project/src/static/index.html"),
        ]
        
        for index_file in possible_paths:
            if index_file.exists():
                return FileResponse(str(index_file))
        
        # If no index.html found, return simple HTML
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>AI Voice Detection API</title></head>
        <body>
            <h1>AI Voice Detection API</h1>
            <p>Status: Running ✅</p>
            <p>API Endpoint: <code>/api/voice-detection</code></p>
            <p>Health Check: <code>/health</code></p>
        </body>
        </html>
        """)
    except Exception as e:
        return {"status": "running", "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "AI Voice Detection API is running",
        "model_loaded": model_loaded
    }

@app.get("/script.js")
async def get_script():
    """Serve script.js"""
    try:
        possible_paths = [
            Path(__file__).parent.parent / "static" / "script.js",
            Path("static") / "script.js",
            Path("/opt/render/project/src/static/script.js"),
        ]
        
        for script_file in possible_paths:
            if script_file.exists():
                return FileResponse(str(script_file), media_type="application/javascript")
        
        raise HTTPException(status_code=404, detail="script.js not found")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/style.css")
async def get_style():
    """Serve style.css"""
    try:
        possible_paths = [
            Path(__file__).parent.parent / "static" / "style.css",
            Path("static") / "style.css",
            Path("/opt/render/project/src/static/style.css"),
        ]
        
        for style_file in possible_paths:
            if style_file.exists():
                return FileResponse(str(style_file), media_type="text/css")
        
        raise HTTPException(status_code=404, detail="style.css not found")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/voice-detection", response_model=DetectionResponse)
async def detect_voice(request: DetectionRequest):
    """Detect if voice is AI-generated or human"""
    
    if not model_loaded or detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading, please try again in a moment"
        )
    
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files as fallback
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")