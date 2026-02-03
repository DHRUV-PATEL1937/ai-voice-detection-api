"""
Download model from Google Drive for deployment
"""
import os
from pathlib import Path
import requests

def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive"""
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def download_and_extract_model():
    """Download model from Google Drive or skip if exists"""
    
    # Check if model already exists (from Git)
    model_path = Path('saved_models/best_model.pth')
    
    if model_path.exists():
        file_size = model_path.stat().st_size
        print(f"✅ Model already exists ({file_size / 1024:.2f} KB)")
        print(f"✅ Model location: {model_path.absolute()}")
        print(f"✅ Skipping download")
        return
    
    # If model doesn't exist, try to download from Google Drive
    model_file_id = os.getenv('MODEL_FILE_ID')
    
    if not model_file_id:
        print("⚠️ MODEL_FILE_ID not set and model not found in repo")
        print("⚠️ Please either:")
        print("   1. Include best_model.pth in saved_models/ directory, OR")
        print("   2. Set MODEL_FILE_ID environment variable in Render")
        print("⚠️ Detection will not work without the model file")
        return
    
    print(f"📦 Downloading model from Google Drive...")
    print(f"   File ID: {model_file_id}")
    
    try:
        # Create saved_models directory if it doesn't exist
        Path('saved_models').mkdir(exist_ok=True)
        
        output_path = 'saved_models/best_model.pth'
        
        # Method 1: Try direct download
        print(f"   Attempting direct download...")
        download_file_from_google_drive(model_file_id, output_path)
        
        # Verify the file was downloaded
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            if file_size > 1000:  # At least 1KB
                print(f"✅ Model downloaded successfully ({file_size / 1024 / 1024:.2f} MB)")
                print(f"✅ Model ready at: {output_path}")
                return
            else:
                print(f"⚠️ Downloaded file is too small ({file_size} bytes)")
                Path(output_path).unlink()
        
        # Method 2: Try alternative download with requests
        print(f"   Trying alternative download method...")
        
        url = f"https://drive.google.com/uc?export=download&id={model_file_id}"
        
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = Path(output_path).stat().st_size
            if file_size > 1000:
                print(f"✅ Model downloaded successfully ({file_size / 1024 / 1024:.2f} MB)")
                print(f"✅ Model ready at: {output_path}")
                return
            else:
                print(f"⚠️ Downloaded file is too small ({file_size} bytes)")
                Path(output_path).unlink()
                raise Exception("Download failed - file too small")
        else:
            raise Exception(f"Download failed with status code {response.status_code}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"\n⚠️ TROUBLESHOOTING STEPS:")
        print(f"   1. Verify Google Drive file is shared publicly")
        print(f"      - Right-click file → Share → 'Anyone with the link' → Viewer")
        print(f"   2. Verify file ID is correct: {model_file_id}")
        print(f"      - File ID is the part after '/d/' in the share link")
        print(f"      - Example: https://drive.google.com/file/d/FILE_ID_HERE/view")
        print(f"   3. Recommended: Include model directly in Git repo (file is only ~577KB)")
        print(f"      - Run: git add saved_models/best_model.pth")
        print(f"      - Run: git commit -m 'Add model file'")
        print(f"      - Run: git push origin deployment")
        print(f"\n⚠️ API will start but voice detection will fail without the model")

if __name__ == "__main__":
    download_and_extract_model()