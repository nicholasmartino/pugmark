import os

from Globals import SECRETS_PATH
from google.cloud import secretmanager, storage


def get_cloud_credentials():
    """Retrieve credentials from Secret Manager."""
    try:
        # Create the Secret Manager client
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name
        name = (
            f"projects/pugmark-448918/secrets/pugmark-service-account/versions/latest"
        )

        # Access the secret
        response = client.access_secret_version(request={"name": name})

        # Return the decoded payload
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Error accessing secret: {e}")
        raise


def get_local_credentials():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
    # path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)
    # PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

    # Configure GCS access (choose one method below)
    # Option 1: If running locally, set credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SECRETS_PATH

    # Add this before any GCS operations
    client = storage.Client.from_service_account_json(SECRETS_PATH)

    # Verify bucket access
    bucket = client.get_bucket("metro-vancouver-regional-district")
    print(f"Bucket exists: {bucket.exists()}")
