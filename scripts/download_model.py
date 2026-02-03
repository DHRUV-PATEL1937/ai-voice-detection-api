"""
Download model from Google Drive on server startup
"""
import os
import zipfile
from pathlib import Path
import gdown

def download_and_extract_model():
    """Download and extract model from Google Drive"""
    
    model_dir = Path('saved_models')
    
    # Check if model already exists
    existing_models = list(model_dir.glob('*/best_model.pth'))
    if existing_models:
        print(f"✅ Model already exists at: {existing_models[0]}")
        return str(existing_models[0])
    
    print("📥 Downloading model from Google Drive...")
    
    # Get file ID from environment variable
    file_id = '1lx0gQTQLodpugBIt26u-_1p1G0GZ9eR1'
    
    if not file_id:
        print("⚠️  MODEL_FILE_ID not set, skipping download")
        return None
    
    # Download from Google Drive
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.zip'
    
    try:
        print(f"   Downloading from Google Drive...")
        gdown.download(url, output, quiet=False, fuzzy=True)
        
        # Extract
        print("📦 Extracting model...")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        # Cleanup
        os.remove(output)
        
        # Verify
        extracted_models = list(model_dir.glob('*/best_model.pth'))
        if extracted_models:
            print(f"✅ Model ready: {extracted_models[0]}")
            return str(extracted_models[0])
        else:
            print("❌ Model file not found after extraction")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    download_and_extract_model()