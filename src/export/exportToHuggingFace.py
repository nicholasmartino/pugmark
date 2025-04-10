import json
import os
import shutil

import tensorflow as tf
from huggingface_hub import HfApi, login

from src.training.Generator import Generator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "template")


def export_and_deploy_to_huggingface(
    export_dir=None, repo_id=None, push_to_hub=False, model_path=None
):
    """
    Export the model in Hugging Face compatible format and optionally deploy to HF Hub.
    Uses the local JSON model instead of GCS checkpoints.

    Args:
        export_dir: Directory to save the exported model
        repo_id: Hugging Face Hub repository ID (e.g., 'username/model-name')
        push_to_hub: Whether to push to Hugging Face Hub
        model_path: Path to the local model JSON file
    """
    # Set default export directory in data folder
    if export_dir is None:
        export_dir = os.path.join("data", "hf_export")

    # Create directory structure
    os.makedirs(export_dir, exist_ok=True)

    # Create and initialize the model
    generator = Generator()
    dummy_input = tf.random.normal([1, 256, 256, 3])
    _ = generator(dummy_input, training=False)

    # Default model path or use provided path
    if model_path is None:
        model_path = os.path.join("data", "model", "model.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model not found at {model_path}")

    # Load and set weights
    with open(model_path, "r") as f:
        weights = json.load(f)

    for layer in generator.layers:
        if layer.name in weights:
            layer.set_weights(weights[layer.name])

    # Save the model in TF SavedModel format
    model_path = os.path.join(export_dir, "tf_model")
    print(f"Saving model to {model_path}")
    tf.saved_model.save(generator, model_path)

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
    repo_url = api.create_repo(
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
        help="Path to the local model.json file (default: data/model/model.json)",
    )

    args = parser.parse_args()

    export_and_deploy_to_huggingface(
        export_dir=args.export_dir,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
        model_path=args.model_path,
    )
