"""
Download and extract model from Google Drive
"""

import os
import zipfile
from pathlib import Path
import gdown


def download_model_from_gdrive():
    """Download model from Google Drive"""

    model_dir = Path("saved_models")

    # Check if model exists
    model_files = list(model_dir.glob("*/best_model.pth"))
    if model_files:
        print("✅ Model already exists")
        return str(model_files[0])

    print("📥 Downloading model from Google Drive...")

    # ✅ Your Google Drive File ID
    file_id = "1lx0gQTQLodpugBIt26u-_1p1G0GZ9eR1"

    # Direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    output = "model.zip"

    try:
        # Download zip
        gdown.download(url, output, quiet=False)

        # Extract zip
        print("📦 Extracting model...")
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(".")

        # Delete zip after extraction
        os.remove(output)

        print("✅ Model downloaded and extracted successfully")

        # Locate model file
        model_files = list(model_dir.glob("*/best_model.pth"))
        if model_files:
            return str(model_files[0])
        else:
            raise Exception("Model file not found after extraction")

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        raise


if __name__ == "__main__":
    download_model_from_gdrive()
