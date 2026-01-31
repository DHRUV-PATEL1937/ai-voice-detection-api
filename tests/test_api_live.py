"""
Test the API with real audio files
"""
import requests
import base64
import json
from pathlib import Path

def test_api_with_file(audio_file_path, language, api_key="sk_test_123456789"):
    """
    Test API with an actual audio file
    
    Args:
        audio_file_path: Path to MP3 file
        language: Language of the audio
        api_key: Your API key
    """
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {audio_file_path}")
    print(f"{'='*60}")
    
    # Read and encode audio file
    try:
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"✅ Audio file loaded ({len(audio_bytes)} bytes)")
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # API endpoint
    url = "http://localhost:8000/api/voice-detection"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    
    # Request body
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    print(f"📤 Sending request to API...")
    
    try:
        # Make request
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"\n📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n{'='*60}")
            print(f"✅ DETECTION RESULT")
            print(f"{'='*60}")
            print(f"Status:        {result['status']}")
            print(f"Language:      {result['language']}")
            print(f"Classification: {result['classification']}")
            print(f"Confidence:    {result['confidenceScore']:.2%}")
            print(f"Explanation:   {result['explanation']}")
            print(f"{'='*60}\n")
            
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")


def test_multiple_files():
    """Test with multiple files from your dataset"""
    
    test_cases = [
        # Format: (file_path, language, expected_label)
        ("dataset/test/human/english", "English", "HUMAN"),
        ("dataset/test/ai_generated/english", "English", "AI_GENERATED"),
        ("dataset/test/human/tamil", "Tamil", "HUMAN"),
        ("dataset/test/ai_generated/tamil", "Tamil", "AI_GENERATED"),
    ]
    
    results = []
    
    for folder, language, expected in test_cases:
        folder_path = Path(folder)
        
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder}")
            continue
        
        # Get first audio file
        audio_files = list(folder_path.glob("*.mp3")) + list(folder_path.glob("*.wav"))
        
        if not audio_files:
            print(f"⚠️  No audio files in: {folder}")
            continue
        
        test_file = audio_files[0]
        
        # Test it
        print(f"\nExpected: {expected}")
        test_api_with_file(test_file, language)
        
        # Add delay to avoid overwhelming API
        import time
        time.sleep(1)


def test_invalid_api_key():
    """Test with invalid API key"""
    print("\n🧪 Testing Invalid API Key...")
    
    url = "http://localhost:8000/api/voice-detection"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "invalid_key_123"
    }
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": "dGVzdA=="
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 401:
        print("✅ Correctly rejected invalid API key")
    else:
        print(f"❌ Unexpected response: {response.status_code}")


def test_health_endpoint():
    """Test health check"""
    print("\n🧪 Testing Health Endpoint...")
    
    response = requests.get("http://localhost:8000/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ API is healthy!")
        print(f"   Status: {result['status']}")
        print(f"   Model Loaded: {result['model_loaded']}")
        print(f"   Supported Languages: {', '.join(result['supported_languages'])}")
    else:
        print(f"❌ Health check failed: {response.status_code}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 AI VOICE DETECTION API - TEST SUITE")
    print("="*60)
    
    # Test 1: Health check
    test_health_endpoint()
    
    # Test 2: Invalid API key
    test_invalid_api_key()
    
    # Test 3: Real detections
    print("\n" + "="*60)
    print("🎯 TESTING REAL VOICE DETECTIONS")
    print("="*60)
    
    test_multiple_files()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE")
    print("="*60)