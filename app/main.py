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
    allow_origins=["*"],  # Allow all origins
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_ready": detector is not None
    }

@app.get("/test")
async def test_page():
    """Simple test page to verify Render is working"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Render Test</title>
    </head>
    <body>
        <h1>✅ Render is serving pages!</h1>
        <p>API URL: <span id="url"></span></p>
        <button onclick="testAPI()">Test API</button>
        <div id="result"></div>
        
        <script>
            document.getElementById('url').textContent = window.location.origin;
            
            async function testAPI() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    document.getElementById('result').innerHTML = 
                        '<p style="color: green;">✅ API is working! ' + JSON.stringify(data) + '</p>';
                } catch (error) {
                    document.getElementById('result').innerHTML = 
                        '<p style="color: red;">❌ API error: ' + error.message + '</p>';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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

# Root route - serve index.html
@app.get("/")
async def root():
    """Serve the frontend"""
    index_path = Path("static/index.html")
    
    if not index_path.exists():
        logger.error(f"❌ index.html not found at: {index_path.absolute()}")
        return HTMLResponse(
            content="""
            <h1>❌ Error: index.html not found</h1>
            <p>Expected location: static/index.html</p>
            <p><a href="/test">Go to test page</a></p>
            """,
            status_code=500
        )
    
    logger.info(f"📄 Serving index.html from: {index_path.absolute()}")
    return FileResponse(index_path)

# Serve static files (CSS, JS, etc.)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Static files mounted at /static/")
except Exception as e:
    logger.error(f"❌ Failed to mount static files: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)