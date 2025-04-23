import os
import shutil

from huggingface_hub import HfApi, login

from src.export.model_utils import find_latest_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "template")


def export_and_deploy_to_huggingface(export_dir=None, repo_id=None):
    """
    Export the saved model in Hugging Face compatible format and deploy to HF Hub.

    Args:
        export_dir: Directory to save the template files (model already cached)
        repo_id: Hugging Face Hub repository ID (e.g., 'username/model-name')
    """
    # Find the already-cached model
    model_path = find_latest_model(use_cache=True)
    if not model_path:
        raise FileNotFoundError("No model found to export")

    # Use model directory as the export directory - add templates directly there
    print(f"Using cached model from: {model_path}")
    copy_templates(model_path)

    # Deploy to Hugging Face Hub
    if not repo_id:
        raise ValueError("Repository ID (repo_id) must be provided")
    deploy_to_hub(model_path, repo_id)


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
    repo_id = "nicholasmartino/pugmark"  # Your Hugging Face repository ID

    # Run the export and deploy function
    export_and_deploy_to_huggingface(repo_id=repo_id)
