import getpass
import json
import os
import shutil

import tensorflow as tf
from huggingface_hub import HfApi, login

from src.training.Exporter import create_generator_model
from src.training.Globals import PATH

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "templates")


def export_and_deploy_to_huggingface(
    checkpoint_dir, export_dir="hf_model", repo_id=None, push_to_hub=False
):
    """
    Export the model in Hugging Face compatible format and optionally deploy to HF Hub.

    Args:
        checkpoint_dir: Directory containing model checkpoints
        export_dir: Local directory to save the exported model
        repo_id: Hugging Face Hub repository ID (e.g., 'username/model-name')
        push_to_hub: Whether to push the model to Hugging Face Hub
    """
    print(f"Looking for latest checkpoint in {checkpoint_dir}")
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if not latest_checkpoint:
        raise ValueError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Found checkpoint: {latest_checkpoint}")

    # Create directory structure
    os.makedirs(export_dir, exist_ok=True)

    # Create and initialize the model
    generator = create_generator_model()

    # Initialize with a dummy input to build the model
    dummy_input = tf.random.normal([1, 256, 256, 3])
    _ = generator(dummy_input)

    # Print model summary
    generator.summary()

    # Setup checkpoint and attempt to restore
    checkpoint = tf.train.Checkpoint(generator=generator)

    try:
        # Restore weights
        print(f"Restoring from: {latest_checkpoint}")
        status = checkpoint.restore(latest_checkpoint).expect_partial()
        print("Restore completed successfully")

        # Save the model in TF SavedModel format (compatible with HF)
        model_path = os.path.join(export_dir, "tf_model")
        print(f"Saving TensorFlow model to {model_path}")
        tf.saved_model.save(generator, model_path)

        # Copy template files to the export directory
        copy_templates(export_dir)

        # Create config.json with model metadata
        create_config(export_dir)

        print(f"Model exported successfully to {export_dir}")

        # Push to Hugging Face Hub if requested
        if push_to_hub:
            if not repo_id:
                raise ValueError(
                    "Repository ID (repo_id) must be provided when push_to_hub is True"
                )

            deploy_to_hub(export_dir, repo_id)

    except Exception as e:
        print(f"Failed to export model: {e}")
        manual_restore(generator, latest_checkpoint, export_dir)
        if push_to_hub and repo_id:
            deploy_to_hub(export_dir, repo_id)


def deploy_to_hub(model_dir, repo_id):
    """
    Deploy the exported model to Hugging Face Hub.

    Args:
        model_dir: Local directory containing the exported model
        repo_id: Hugging Face Hub repository ID (e.g., 'username/model-name')
    """
    print(f"\n=== DEPLOYING TO HUGGING FACE HUB ===")
    print(f"Target repository: {repo_id}")

    # Authenticate with Hugging Face
    authenticate_huggingface()

    # Create or ensure repository exists
    try:
        api = HfApi()
        repo_url = api.create_repo(
            repo_id=repo_id,
            private=False,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        print(f"Repository ready: {repo_url}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        raise

    # Upload model files to Hugging Face Hub
    print(f"Uploading files from {model_dir} to {repo_id}...")
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Upload model files",
        )
        print(f"Successfully deployed model to Hugging Face Hub: {repo_id}")
        print(f"Model Space URL: https://huggingface.co/spaces/{repo_id}")
        print(
            f"API Endpoint URL: https://api-inference.huggingface.co/models/{repo_id}"
        )
    except Exception as e:
        print(f"Error uploading model files: {e}")
        raise


def authenticate_huggingface():
    """Authenticate with Hugging Face Hub."""
    try:
        # Try to use cached token
        api = HfApi()
        user_info = api.whoami()
        print(f"Already logged in as: {user_info['name']}")
    except:
        # If token not found or invalid, prompt for login
        print("Hugging Face Hub authentication required")
        token = getpass.getpass(
            "Enter your Hugging Face Hub token (from https://huggingface.co/settings/tokens): "
        )
        login(token=token)
        print("Authentication successful!")


def manual_restore(model, checkpoint_path, export_dir):
    """Manually restore variables from checkpoint when direct restore fails."""
    print("Attempting manual variable extraction...")
    reader = tf.train.load_checkpoint(checkpoint_path)

    for var in model.trainable_variables:
        var_name = var.name.replace(":0", "")
        checkpoint_name = f"generator/{var_name}"

        try:
            if reader.has_tensor(checkpoint_name):
                tensor_value = reader.get_tensor(checkpoint_name)
                var.assign(tensor_value)
                print(f"Loaded: {var.name}")
            else:
                print(f"Missing: {checkpoint_name}")
        except Exception as ex:
            print(f"Error loading {var.name}: {ex}")

    # Save the model after manual extraction
    model_path = os.path.join(export_dir, "tf_model")
    print(f"Saving manually restored model to {model_path}")
    tf.saved_model.save(model, model_path)

    # Copy template files to the export directory
    copy_templates(export_dir)

    # Create config.json with model metadata
    create_config(export_dir)

    print(f"Model exported successfully to {export_dir}")


def copy_templates(export_dir):
    """Copy template files to the export directory."""
    template_files = {
        "README.md": "README.md",
        "app.py": "app.py",
        "requirements.txt": "requirements.txt",
    }

    for template_file, target_file in template_files.items():
        template_path = os.path.join(TEMPLATE_DIR, template_file)
        target_path = os.path.join(export_dir, target_file)

        if os.path.exists(template_path):
            shutil.copy2(template_path, target_path)
            print(f"Copied {template_file} to {target_path}")
        else:
            print(f"Warning: Template file {template_path} not found")


def create_config(export_dir):
    """Create a config.json file with model metadata."""
    config = {
        "model_type": "pix2pix",
        "task": "image-to-image",
        "framework": "tensorflow",
        "architecture": "unet",
        "input_size": [256, 256, 3],
        "output_size": [256, 256, 3],
        "tags": ["satellite-imagery", "building-footprints", "geospatial"],
        "license": "mit",
        "datasets": ["custom-vancouver-satellite"],
    }

    config_path = os.path.join(export_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Created {config_path}")


if __name__ == "__main__":
    # To run this script from the project root:
    # python -m src.export.ExportToHF --push-to-hub --repo-id your-username/model-name
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
        "--checkpoint-dir",
        type=str,
        default=f"{PATH}/footprints/checkpoint",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="hf_model",
        help="Local directory to save the exported model",
    )

    args = parser.parse_args()

    # Run export and optional deployment
    export_and_deploy_to_huggingface(
        checkpoint_dir=args.checkpoint_dir,
        export_dir=args.export_dir,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
    )
