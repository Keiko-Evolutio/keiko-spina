# backend/storage/azure_blob_storage/__init__.py
"""Azure Blob Storage Management."""

from .azure_blob_storage import close_storage_clients, get_storage_client

__all__ = ["close_storage_clients", "get_storage_client"]
