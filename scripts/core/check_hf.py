from huggingface_hub import HfApi
try:
    user_info = HfApi().whoami()
    print(f"Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Not logged in or error: {e}")
