#!/usr/bin/env python3
"""
Monitor a Vertex AI training job and stream logs.
Usage: python monitor_vertex_job.py JOB_ID
"""

import argparse
import os
import sys
import time
from datetime import datetime

from google.cloud import aiplatform, logging


def setup_args():
    parser = argparse.ArgumentParser(description="Monitor a Vertex AI training job")
    parser.add_argument("job_id", help="The Vertex AI job ID to monitor")
    parser.add_argument(
        "--project",
        default=os.environ.get("GCP_PROJECT_ID"),
        help="Google Cloud project ID",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("REGION", "us-central1"),
        help="Google Cloud region",
    )
    return parser.parse_args()


def get_job_status(job_id, project, region):
    """Get the status of a Vertex AI custom job."""
    aiplatform.init(project=project, location=region)

    # Get the job
    job = aiplatform.CustomJob.get(
        resource_name=f"projects/{project}/locations/{region}/customJobs/{job_id}"
    )

    # Get the job state
    state = job.state

    return {
        "state": state,
        "create_time": job.create_time,
        "start_time": job.start_time,
        "update_time": job.update_time,
        "end_time": job.end_time,
        "error": job.error,
        "display_name": job.display_name,
        "job_spec": job.job_spec,
    }


def stream_logs(job_id, project):
    """Stream logs from a Vertex AI custom job."""
    logging_client = logging.Client(project=project)

    # Define the filter for the logs
    filter_str = (
        f'resource.type="aiplatform.googleapis.com/CustomJob" '
        f'resource.labels.custom_job_id="{job_id}" '
        f"severity>=INFO"
    )

    # Get the logger
    logger = logging_client.logger("aiplatform")

    # Get the last timestamp we've seen
    last_timestamp = datetime.utcnow()

    print(f"Streaming logs for job {job_id}...")
    print("-" * 80)

    try:
        while True:
            # Get entries
            entries = list(
                logger.list_entries(
                    filter_=filter_str, order_by="timestamp asc", page_size=100
                )
            )

            # Print new entries
            for entry in entries:
                entry_time = entry.timestamp.replace(tzinfo=None)
                if entry_time > last_timestamp:
                    print(f"[{entry.timestamp}] {entry.payload}")
                    last_timestamp = entry_time

            # Sleep before checking again
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nLog streaming stopped.")


def main():
    args = setup_args()

    if not args.project:
        print(
            "Error: GCP_PROJECT_ID environment variable or --project argument is required"
        )
        sys.exit(1)

    # Get initial job status
    try:
        status = get_job_status(args.job_id, args.project, args.region)
        print(f"Job ID: {args.job_id}")
        print(f"Display Name: {status['display_name']}")
        print(f"State: {status['state']}")
        print(f"Created: {status['create_time']}")
        print(f"Started: {status['start_time']}")
        print(f"Last Updated: {status['update_time']}")

        if status["error"]:
            print(f"Error: {status['error']}")

        print("\nStreaming logs (Ctrl+C to stop)...")
        print("-" * 80)

        # Stream logs
        stream_logs(args.job_id, args.project)

    except Exception as e:
        print(f"Error monitoring job: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
