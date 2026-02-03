"""
Download model from Google Drive for deployment
"""
import gdown
import zipfile
from pathlib import Path
import os

def download_and_extract_model():
    """Download model from Google Drive"""
    
    model_file_id = os.getenv('MODEL_FILE_ID')
    
    if not model_file_id:
        print("⚠️ MODEL_FILE_ID not set, skipping download")
        return
    
    print(f"📦 Downloading model from Google Drive...")
    print(f"   File ID: {model_file_id}")
    
    # Download URL
    url = f"https://drive.google.com/uc?id={model_file_id}"
    
    # Download to temp file
    output = "model.zip"
    
    try:
        gdown.download(url, output, quiet=False)
        print(f"✅ Downloaded to {output}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise
    
    # Extract
    try:
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✅ Extracted model files")
        
        # Remove zip
        os.remove(output)
        print(f"✅ Cleaned up zip file")
        
        # Verify model exists
        if Path('best_model.pth').exists():
            print(f"✅ Model ready at: best_model.pth")
        elif Path('saved_models/best_model.pth').exists():
            print(f"✅ Model ready at: saved_models/best_model.pth")
        else:
            raise FileNotFoundError("Model file not found after extraction")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        raise

if __name__ == "__main__":
    download_and_extract_model()