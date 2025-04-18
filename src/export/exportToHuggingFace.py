import os
import shutil

import tensorflow as tf
from huggingface_hub import HfApi, login

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "template")


def export_and_deploy_to_huggingface(
    export_dir=None, repo_id=None, push_to_hub=False, model_path=None
):
    """
    Export the saved model in Hugging Face compatible format and optionally deploy to HF Hub.

    Args:
        export_dir: Directory to save the exported model
        repo_id: Hugging Face Hub repository ID (e.g., 'username/model-name')
        push_to_hub: Whether to push to Hugging Face Hub
        model_path: Path to the saved model directory
    """
    # Set default export directory in data folder
    if export_dir is None:
        export_dir = os.path.join("data", "hf_export")

    # Create directory structure
    os.makedirs(export_dir, exist_ok=True)

    # Use the model path from the find_latest_model function
    if model_path is None:
        from src.export.simple_test import find_latest_model

        model_path = find_latest_model(use_cache=True)

    if not model_path or not tf.io.gfile.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at {model_path}")

    # Copy the saved model to the export directory
    tf_model_path = os.path.join(export_dir, "tf_model")
    print(f"Copying saved model from {model_path} to {tf_model_path}")

    # Remove existing model if present
    if os.path.exists(tf_model_path):
        shutil.rmtree(tf_model_path)

    # Copy the model files
    for item in tf.io.gfile.listdir(model_path):
        src_path = os.path.join(model_path, item)
        dst_path = os.path.join(tf_model_path, item)

        if tf.io.gfile.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    # Copy template files
    copy_templates(export_dir)
    print(f"Model exported successfully to {export_dir}")

    # Push to Hugging Face Hub if requested
    if push_to_hub:
        if not repo_id:
            raise ValueError(
                "Repository ID (repo_id) must be provided when push_to_hub is True"
            )
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
        commit_message="Upload model files",
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Export and deploy TensorFlow model to Hugging Face Hub"
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push the model to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Hugging Face Hub repository ID (username/model-name)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        help="Local directory to save the exported model (default: data/hf_export)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the saved model directory",
    )

    args = parser.parse_args()

    export_and_deploy_to_huggingface(
        export_dir=args.export_dir,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
        model_path=args.model_path,
    )
