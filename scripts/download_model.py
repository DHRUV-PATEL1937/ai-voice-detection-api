"""
Download model from Google Drive for deployment
"""
import os
import requests
from pathlib import Path

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
    """Download model from Google Drive"""
    
    model_file_id = os.getenv('MODEL_FILE_ID')
    
    if not model_file_id:
        print("⚠️ MODEL_FILE_ID not set, skipping download")
        return
    
    print(f"📦 Downloading model from Google Drive...")
    print(f"   File ID: {model_file_id}")
    
    try:
        # Create saved_models directory if it doesn't exist
        Path('saved_models').mkdir(exist_ok=True)
        
        # Download directly to the final location
        output_path = 'saved_models/best_model.pth'
        
        # Try direct download first
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
                print(f"⚠️ Downloaded file is too small ({file_size} bytes), might be an error page")
                Path(output_path).unlink()
        
        # If direct download failed, try alternative method
        print(f"   Trying alternative download method...")
        
        # Alternative: Use requests with direct link
        url = f"https://drive.google.com/uc?export=download&id={model_file_id}"
        
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = Path(output_path).stat().st_size
            print(f"✅ Model downloaded successfully ({file_size / 1024 / 1024:.2f} MB)")
            print(f"✅ Model ready at: {output_path}")
        else:
            raise Exception(f"Download failed with status code {response.status_code}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"⚠️ Please check:")
        print(f"   1. Is the Google Drive link publicly accessible?")
        print(f"   2. Is the file ID correct: {model_file_id}")
        print(f"   3. Try sharing the file with 'Anyone with the link can view'")
        raise

if __name__ == "__main__":
    download_and_extract_model()