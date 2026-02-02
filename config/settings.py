"""
Application Configuration
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "saved_models" / "human_authenticator_20260201_144346" / "best_model.pth"

# API Settings
API_TITLE = "AI Voice Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Detect AI-generated vs Human voices in Tamil, English, Hindi, Malayalam, and Telugu"

# Supported languages
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# API Keys (In production, use environment variables or database)
VALID_API_KEYS = {
    "sk_test_123456789": {
        "user": "test_user",
        "tier": "free",
        "active": True
    },
    "sk_live_987654321": {
        "user": "production_user", 
        "tier": "premium",
        "active": True
    }
}

# Model Settings
TARGET_SR = 16000
TARGET_DURATION = 5.0
NUM_ACOUSTIC_FEATURES = 50