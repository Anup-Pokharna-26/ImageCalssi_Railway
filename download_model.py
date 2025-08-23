import os
import gdown
import numpy as np

# Configuration
# https://drive.google.com/file/d/10Y8JsMi6GjiYNm9s1c2zzEXGH02mIFtC/view?usp=sharing
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=10Y8JsMi6GjiYNm9s1c2zzEXGH02mIFtC"  # Replace with your model's shared link
# Update this URL to point to a specific file, not a folder
CLASS_NAMES_URL = "https://drive.google.com/uc?id=1yrWuqmKesWPIhxbEzX07lida-BbW9-QF"  # Replace with your class names file link
COLAB_NOTEBOOK_URL = "https://colab.research.google.com/drive/14viVNyKRsvyJyntQhmnOaAyaY0IjOthh?usp=sharing"  # Replace with your Colab notebook link

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)

def download_model():
    """Download the model from Google Drive using gdown."""
    model_path = 'model/best_model.keras'
    print(f"Downloading model from Google Drive...")
    try:
        # Try to download with gdown
        gdown.download(MODEL_DRIVE_URL, model_path, quiet=False)
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            print(f"Model downloaded successfully to {model_path}")
            return model_path
        else:
            raise Exception("Downloaded file is empty or doesn't exist")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please check:")
        print("1. The Google Drive link is correct and publicly accessible")
        print("2. The file ID in the URL is correct")
        print("3. Your internet connection is working")
        raise e

def download_class_names():
    """Download the class names file from Google Drive."""
    class_names_path = 'model/class_names.txt'
    print(f"Downloading class names from Google Drive...")
    print(f"URL: {CLASS_NAMES_URL}")
    
    try:
        gdown.download(CLASS_NAMES_URL, class_names_path, quiet=False)
        
        if os.path.exists(class_names_path) and os.path.getsize(class_names_path) > 0:
            # Read class names with explicit UTF-8 encoding
            with open(class_names_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines()]
            
            print(f"Found {len(class_names)} classes: {class_names}")
            return class_names
        else:
            raise Exception("Downloaded class names file is empty or doesn't exist")
    
    except Exception as e:
        print(f"Error downloading class names: {e}")
        print("Please check:")
        print("1. The Google Drive link is correct and publicly accessible")
        print("2. The file ID in the URL is correct")
        print("3. Your internet connection is working")
        raise e

def get_model_timestamp():
    """Get the timestamp of the model file for checking updates."""
    model_path = 'model/best_model.keras'
    if os.path.exists(model_path):
        return os.path.getmtime(model_path)
    return 0

if __name__ == '__main__':
    # Download model and class names
    download_model()
    download_class_names()