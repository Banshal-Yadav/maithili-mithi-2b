import os
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parents[2]

def download_model():
    model_id = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    local_dir = str(BASE_DIR / "base_models" / "llama-3.2-3b")
    
    print(f"Downloading {model_id} to {local_dir}...")
    
    # Ensure the directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision="main"
        )
        print(f"Download complete! Model saved in {local_dir}")
    except Exception as e:
        import traceback
        print(f"An error occurred during download: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    download_model()
