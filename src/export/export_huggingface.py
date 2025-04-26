import os
import shutil

from huggingface_hub import HfApi, login

# No longer need this import since we're using a fixed path
# from src.export.model_utils import find_latest_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "template")
MODEL_PATH = os.path.join("cache", "model_cache")  # Fixed model path


def export_and_deploy_to_huggingface(export_dir=None, repo_id=None):
    """
    Export the saved model in Hugging Face compatible format and deploy to HF Hub.

    Args:
        export_dir: Directory to save the exported files
        repo_id: Hugging Face Hub repository ID (e.g., 'username/model-name')
    """
    # Set default export directory if not provided
    if export_dir is None:
        export_dir = os.path.join("cache", "hf_export")

    # Create and clean the export directory
    os.makedirs(export_dir, exist_ok=True)

    # Use the fixed model path instead of finding it
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}")

    print(f"Using model from: {model_path}")
    print(f"Model files: {os.listdir(model_path)}")

    # Create the tf_model directory in the export directory
    tf_model_path = os.path.join(export_dir, "tf_model")

    # Remove existing tf_model directory if it exists
    if os.path.exists(tf_model_path):
        shutil.rmtree(tf_model_path)

    # Create the tf_model directory
    os.makedirs(tf_model_path, exist_ok=True)

    # Copy all files from model_path to tf_model_path
    print(f"Copying model files from {model_path} to {tf_model_path}")
    for item in os.listdir(model_path):
        src = os.path.join(model_path, item)
        dst = os.path.join(tf_model_path, item)

        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Verify the copied model structure
    copied_files = os.listdir(tf_model_path)
    print(f"Files copied to {tf_model_path}: {copied_files}")

    # Copy template files to the export directory
    copy_templates(export_dir)
    print(f"Export preparation complete at {export_dir}")

    # Deploy to Hugging Face Hub
    if not repo_id:
        raise ValueError("Repository ID (repo_id) must be provided")
    deploy_to_hub(export_dir, repo_id)


def deploy_to_hub(model_dir, repo_id):
    """Deploy the exported model to Hugging Face Hub."""
    print(f"\n=== DEPLOYING TO HUGGING FACE HUB ===")
    print(f"Target repository: {repo_id}")

    # Authenticate with Hugging Face
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"Already logged in as: {user_info['name']}")
    except:
        print("Hugging Face Hub authentication required")
        token = input("Enter your Hugging Face Hub token: ")
        login(token=token)

    # Create or ensure repository exists
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        private=False,
        exist_ok=True,
        repo_type="space",
        space_sdk="gradio",
    )

    # Upload model files
    print(f"Uploading files from {model_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="space",
        commit_message="Upload footprints pix2pix model",
    )
    print(f"Successfully deployed to: https://huggingface.co/spaces/{repo_id}")


def copy_templates(export_dir):
    """Copy template files to the export directory."""
    for template_file in ["README.md", "app.py", "requirements.txt"]:
        src = os.path.join(TEMPLATE_DIR, template_file)
        dst = os.path.join(export_dir, template_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {template_file}")


if __name__ == "__main__":
    # Hard-coded configuration
    export_dir = "cache/hf_export"  # Directory for export
    repo_id = "nicholasmartino/pugmark"  # Your Hugging Face repository ID

    # Run the export and deploy function
    export_and_deploy_to_huggingface(
        export_dir=export_dir,
        repo_id=repo_id,
    )
