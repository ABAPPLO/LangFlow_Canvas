"""Upload File component - uploads files and returns paths/metadata without parsing content."""

from __future__ import annotations

from copy import deepcopy

from lfx.base.data.base_file import BaseFileComponent
from lfx.io import Output
from lfx.schema.data import Data
from lfx.services.deps import get_settings_service


class UploadFileComponent(BaseFileComponent):
    """Upload files and return file paths and metadata without parsing content."""

    display_name = "Upload File"
    description = "Uploads files and returns file paths and metadata without parsing content."
    icon = "upload"
    name = "UploadFile"

    # Accept all file types since we don't parse content
    VALID_EXTENSIONS: list[str] = []

    _base_inputs = deepcopy(BaseFileComponent.get_base_inputs())

    inputs = [*_base_inputs]

    outputs = [
        Output(display_name="File Path", name="filepath", method="load_files_path"),
        Output(display_name="Files", name="files", method="load_files"),
        Output(display_name="Table", name="table", method="load_files_table"),
        Output(display_name="Data", name="data", method="load_files_data"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Clear file type restriction - empty list means accept all types
        self.get_base_inputs()[0].file_types = []
        self.get_base_inputs()[0].info = "Accept all file types."

    def _filter_and_mark_files(self, files):
        """Accept all file types without extension validation."""
        settings = get_settings_service().settings
        is_cloud_storage = settings.storage_type in ("s3", "cos", "oss")
        return [f for f in files if is_cloud_storage or f.path.is_file()]

    def process_files(self, file_list):
        """Skip content parsing - only populate file metadata."""
        for base_file in file_list:
            path_str = str(base_file.path)
            data = Data(
                data={
                    "url": path_str,
                    "file_name": base_file.path.name,
                }
            )
            base_file.data = [data]
        return file_list

    def load_files_table(self) -> list[Data]:
        """Return file metadata as a list of Data objects (table format)."""
        return self.load_files_core()

    def load_files_data(self) -> Data:
        """Return all file metadata as a single Data object."""
        data_list = self.load_files_core()
        if not data_list:
            return Data()
        combined = {}
        for i, d in enumerate(data_list):
            combined[f"file_{i}"] = d.data
        return Data(data=combined)
