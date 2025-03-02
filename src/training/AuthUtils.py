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

        # Check if we have a credentials file
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.exists(creds_path):
            print(f"Using credentials file: {creds_path}")

            # For TensorFlow to use the same credentials
            os.environ["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] = creds_path

            # Use the same credentials for our client
            return storage.Client.from_service_account_json(creds_path)

        # If using Workload Identity Federation (access_token)
        access_token = os.getenv("GCP_ACCESS_TOKEN")
        if access_token:
            print("Using access token from GitHub Actions")
            # This environment variable is critical for TensorFlow GFile operations
            os.environ["GOOGLE_CLOUD_ACCESS_TOKEN"] = access_token

        # Get the storage client
        client = storage.Client()

        return client
    except Exception as e:
        print(f"Error creating storage client for GitHub Actions: {e}")
        raise


def setup_tensorflow_auth():
    """
    Sets up authentication for TensorFlow's GFile operations.
    This ensures TensorFlow can access GCS using the same credentials.
    """
    try:
        # Check if GOOGLE_APPLICATION_CREDENTIALS is set
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.exists(creds_path):
            print(f"Using credentials from: {creds_path}")
            # TensorFlow should pick these up automatically
            return

        # If we're in GitHub Actions, we need to use the token directly
        if os.getenv("GITHUB_ACTIONS") == "true":
            print("Setting up TensorFlow auth for GitHub Actions")
            # Try to get the access token set by the github-actions/auth step
            access_token = os.getenv("GCP_ACCESS_TOKEN")
            if access_token:
                # Set environment variable that TensorFlow's GFile operations can use
                os.environ["GOOGLE_CLOUD_ACCESS_TOKEN"] = access_token
                print("Set GOOGLE_CLOUD_ACCESS_TOKEN for TensorFlow GFile operations")
    except Exception as e:
        print(f"Warning: Failed to set up TensorFlow authentication: {e}")


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
        client = storage.Client()
        setup_tensorflow_auth()
        return client

    # Set up TensorFlow's authentication
    setup_tensorflow_auth()

    return storage.Client.from_service_account_json(secret_path)
