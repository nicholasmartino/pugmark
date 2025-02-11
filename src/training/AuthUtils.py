import os

from dotenv import load_dotenv
from google.cloud import secretmanager, storage


def get_cloud_credentials():
    """Retrieve credentials from Secret Manager."""
    try:
        # Load environment variables
        load_dotenv()
        project_id = os.getenv("GCP_PROJECT_ID")
        secret_name = os.getenv("GCP_SECRET_NAME")

        # Create the Secret Manager client
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"

        # Access the secret
        response = client.access_secret_version(request={"name": name})

        # Return the decoded payload
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Error accessing secret: {e}")
        raise


def get_local_credentials():
    # _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
    # path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)
    # PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

    # Configure GCS access (choose one method below)
    # Option 1: If running locally, set credentials
    secret_path = os.getenv("GCP_SECRET_PATH")

    # Add this before any GCS operations
    load_dotenv()
    client = storage.Client.from_service_account_json(secret_path)

    # Verify bucket access
    bucket = client.get_bucket("metro-vancouver-regional-district")
    print(f"Bucket exists: {bucket.exists()}")
