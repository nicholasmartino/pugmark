import os

from dotenv import load_dotenv
from google.cloud import secretmanager, storage


def auth_client_from_cloud():
    """Retrieve credentials from Secret Manager and access GCS bucket."""
    try:
        # Load environment variables
        load_dotenv()
        project_id = os.getenv("GCP_PROJECT_ID")
        secret_name = os.getenv("GCP_SECRET_NAME")

        # Create the Secret Manager client
        secret_manager = secretmanager.SecretManagerServiceClient()

        # Build the resource name
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"

        # Access the secret
        response = secret_manager.access_secret_version(request={"name": name})

        return storage.Client(credentials=response.payload.data.decode("UTF-8"))
    except Exception as e:
        print(f"Error accessing secret or bucket: {e}")
        raise


def auth_client_locally():
    # _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
    # path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)
    # PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

    # Configure GCS access (choose one method below)
    # Option 1: If running locally, set credentials
    secret_path = os.getenv("GCP_SECRET_PATH")

    # Add this before any GCS operations
    load_dotenv()
    return storage.Client.from_service_account_json(secret_path)
