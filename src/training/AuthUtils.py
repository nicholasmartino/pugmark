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


def auth_client_locally():
    # Configure GCS access (choose one method below)
    # Option 1: If running locally, set credentials
    secret_path = os.getenv("GCP_SECRET_PATH")

    # Add this before any GCS operations
    load_dotenv()
    return storage.Client.from_service_account_json(secret_path)
