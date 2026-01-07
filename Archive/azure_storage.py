"""
Azure Storage utilities for CMS data processing (local fallback)
"""
import os
import io
import logging
import time
import pandas as pd
from azure.storage.blob import BlobServiceClient, ContainerClient, ContentSettings
from azure.core.exceptions import ServiceRequestError, ServiceResponseError, HttpResponseError
import mimetypes

logger = logging.getLogger('cms_function.azure_storage')

class AzureStorageClient:
    def __init__(self, connection_string=None):
        """Initialize Azure Storage client with connection string"""
        if connection_string is None:
            connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            raise RuntimeError(
                "AZURE_STORAGE_CONNECTION_STRING is not set. Please set it in your environment or pass it to AzureStorageClient()."
            )
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = "cmsfiles"
    
    def _with_retries(self, fn, op_desc: str, max_attempts: int = 3, base_delay: float = 1.0):
        """Run fn() with simple exponential backoff on transient Azure errors."""
        attempt = 1
        while True:
            try:
                return fn()
            except (ServiceRequestError, ServiceResponseError, HttpResponseError) as ex:
                if attempt >= max_attempts:
                    logger.error(f"{op_desc} failed after {attempt} attempts: {ex}")
                    raise
                sleep = base_delay * (2 ** (attempt - 1))
                logger.warning(f"{op_desc} failed (attempt {attempt}/{max_attempts}): {ex}. Retrying in {sleep:.1f}s")
                time.sleep(sleep)
                attempt += 1
        
    def list_blobs(self, prefix=None):
        """List all blobs in the container with optional prefix filter"""
        container_client = self.blob_service_client.get_container_client(self.container_name)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blobs]
    
    def read_csv_from_blob(self, blob_name, **kwargs):
        """Read a CSV file from Azure Blob Storage into a pandas DataFrame"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            # Download blob content
            download_stream = self._with_retries(lambda: blob_client.download_blob(), f"download_blob({blob_name})")
            content = self._with_retries(lambda: download_stream.readall(), f"read_blob_bytes({blob_name})")
            
            # Create a file-like object from the content
            file_like_object = io.BytesIO(content)
            
            # Read the CSV into a pandas DataFrame
            df = pd.read_csv(file_like_object, **kwargs)
            
            logger.info(f"Successfully read CSV from blob: {blob_name}")
            return df
        
        except Exception as e:
            logger.error(f"Error reading CSV from blob {blob_name}: {str(e)}")
            raise
    
    def write_dataframe_to_blob(self, df, blob_name, **kwargs):
        """Write a pandas DataFrame to a CSV file in Azure Blob Storage.

        - Uploads bytes with explicit CSV content-type
        - Sets metadata to force lastModified/etag update even if content is identical
        - Performs read-after-write to log size and lastModified for verification
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)

            # Convert DataFrame to CSV text then encode to bytes
            if 'index' not in kwargs:
                kwargs['index'] = False
            csv_text = df.to_csv(**kwargs)
            data = csv_text.encode('utf-8')

            content_settings = ContentSettings(content_type='text/csv; charset=utf-8')
            self._with_retries(lambda: blob_client.upload_blob(data, overwrite=True, content_settings=content_settings), f"upload_blob({blob_name})")

            # Force metadata update to guarantee a lastModified change each run
            try:
                from datetime import datetime, timezone
                run_ts = datetime.now(timezone.utc).isoformat()
                run_id = os.environ.get('WEBSITE_INSTANCE_ID', '')[:12]
                meta = {"last_run": run_ts, "run_id": run_id}
                self._with_retries(lambda: blob_client.set_blob_metadata(metadata=meta), f"set_metadata({blob_name})")
            except Exception as meta_ex:
                logger.warning(f"Blob metadata update failed for {blob_name}: {meta_ex}")

            # Read-after-write verify
            try:
                props = self._with_retries(lambda: blob_client.get_blob_properties(), f"get_blob_properties({blob_name})")
                sz = props.size if hasattr(props, 'size') else props.get('size', None)
                lm = props.last_modified if hasattr(props, 'last_modified') else props.get('last_modified', None)
                etag = props.etag if hasattr(props, 'etag') else props.get('etag', None)
                logger.info(
                    f"Visuals write verify: {blob_name} size={sz} lastModified={lm} etag={etag} bytes_written={len(data):,}"
                )
            except Exception as ver_ex:
                logger.warning(f"Post-write properties fetch failed for {blob_name}: {ver_ex}")

            logger.info(f"Successfully wrote DataFrame to blob: {blob_name} (bytes={len(data):,})")

        except Exception as e:
            logger.error(f"Error writing DataFrame to blob {blob_name}: {str(e)}")
            raise

    def blob_exists(self, blob_name: str) -> bool:
        """Check whether a blob exists in the container."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)
            return self._with_retries(lambda: blob_client.exists(), f"blob_exists({blob_name})")
        except Exception as e:
            logger.error(f"Error checking existence for blob {blob_name}: {e}")
            return False

    def upload_file(self, local_path: str, blob_name: str, content_type: str | None = None, overwrite: bool = True):
        """Upload a local file to the container as blob_name."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)
            if content_type is None:
                guessed, _ = mimetypes.guess_type(local_path)
                content_type = guessed or 'application/octet-stream'
            with open(local_path, 'rb') as f:
                file_bytes = f.read()
            cs = ContentSettings(content_type=content_type)
            self._with_retries(lambda: blob_client.upload_blob(file_bytes, overwrite=overwrite, content_settings=cs), f"upload_file({blob_name})")
            logger.info(f"Uploaded {local_path} to blob {blob_name}")
        except Exception as e:
            logger.error(f"Error uploading file to blob {blob_name}: {e}")
            raise

