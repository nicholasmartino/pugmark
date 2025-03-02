import os

from dotenv import load_dotenv
from google.cloud import storage


def auth_client_from_cloud():
    """Get storage client when running in Google Cloud environment."""
    try:
        project_id = "pugmark-448918"
        print(f"Creating storage client for project {project_id}")

        # When running on Cloud Run, use default credentials
        return storage.Client(project=project_id)
    except Exception as e:
        print(f"Error creating storage client: {e}")
        raise


def auth_client_github_actions():
    """Get storage client when running in GitHub Actions environment."""
    try:
        print(
            "Creating storage client using Application Default Credentials (GitHub Actions)"
        )
        # Use application default credentials set up by github-actions/auth
        return storage.Client()
    except Exception as e:
        print(f"Error creating storage client for GitHub Actions: {e}")
        raise


def auth_client_locally():
    """Get storage client when running in local environment."""
    # Check if we're in GitHub Actions environment
    if os.getenv("GITHUB_ACTIONS") == "true":
        return auth_client_github_actions()

    # Configure GCS access for local development
    # Option 1: If running locally, set credentials
    secret_path = os.getenv("GCP_SECRET_PATH")

    # Add this before any GCS operations
    load_dotenv()

    if not secret_path:
        print(
            "Warning: GCP_SECRET_PATH environment variable not set. Authentication may fail."
        )
        # Try using application default credentials as fallback
        return storage.Client()

    return storage.Client.from_service_account_json(secret_path)
