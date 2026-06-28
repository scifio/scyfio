"""A SCIFIO-based scientific image reader for Python."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scyfio")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._core_metadata import CoreMetadata, OMEShape
from ._image_file import ImageFile
from ._imread import imread, open_ome_zarr_group, open_zarr_array
from ._lazy_array import LazyImageArray
from ._series import Series

__all__ = [
    "CoreMetadata",
    "ImageFile",
    "LazyImageArray",
    "OMEShape",
    "Series",
    "imread",
    "open_ome_zarr_group",
    "open_zarr_array",
]
