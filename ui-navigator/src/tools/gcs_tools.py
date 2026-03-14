from datetime import datetime
from google.cloud import storage
from src.config import config

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = storage.Client(project=config.GCP_PROJECT_ID)
    return _client


def upload_screenshot(
    screenshot_bytes: bytes,
    screen_id: str,
    run_id: str = "",
    suffix: str = "",
) -> str:
    """Upload a screenshot to GCS and return the GCS URI.

    Args:
        screenshot_bytes: Raw PNG bytes.
        screen_id: Screen ID used for path organization.
        run_id: Run ID for scoping screenshots per exploration run.
        suffix: Optional suffix (e.g. 'after_action_xyz').

    Returns:
        GCS URI string (gs://bucket/path) or empty string on failure.
    """
    try:
        client = _get_client()
        bucket = client.bucket(config.GCS_BUCKET)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name = f"{screen_id}_{ts}"
        if suffix:
            name += f"_{suffix}"

        # Scope under run_id if provided
        if run_id:
            blob_path = f"runs/{run_id}/screenshots/{screen_id}/{name}.png"
        else:
            blob_path = f"screenshots/{screen_id}/{name}.png"

        blob = bucket.blob(blob_path)
        blob.upload_from_string(screenshot_bytes, content_type="image/png")

        return f"gs://{config.GCS_BUCKET}/{blob_path}"
    except Exception as e:
        print(f"WARNING: GCS upload failed: {e}")
        return ""
