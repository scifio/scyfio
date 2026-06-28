from __future__ import annotations

import os
import sys
import warnings
import weakref
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal, overload

import jpype
import numpy as np
from ome_types import OME

from scyfio._core_metadata import CoreMetadata
from scyfio._series import Series

from . import _utils
from ._java_stuff import get_scifio, jtype_to_python
from ._jimports import jimport

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    import dask.array
    import xarray as xr
    from typing_extensions import Self

    # SCIFIO's io.scif.Reader has no type stub in ./typings; treat it opaquely.
    IFormatReader = Any

    from scyfio._lazy_array import LazyBioArray
    from scyfio._zarr import BFOmeZarrStore
    from scyfio._zarr._array_store import BFArrayStore


@dataclass(frozen=True)
class ReaderInfo:
    """Information about a Bio-Formats reader class.

    Attributes
    ----------
    format : str
        Human-readable format name (e.g., "Nikon ND2").
    suffixes : tuple[str, ...]
        Supported file extensions (e.g., ("nd2", "jp2")).
    class_name : str
        Full Java class name (e.g., "ND2Reader").
    is_gpl : bool
        Whether this reader requires GPL license (True) or is BSD (False).
    """

    format: str
    suffixes: tuple[str, ...]
    class_name: str
    is_gpl: bool


# by default, .bfmemo files will go into the same directory as the file.
# users can override this with BIOFORMATS_MEMO_DIR env var
BIOFORMATS_MEMO_DIR: Path | None = None
_BFDIR = os.getenv("BIOFORMATS_MEMO_DIR")
if _BFDIR:
    BIOFORMATS_MEMO_DIR = Path(_BFDIR).expanduser().absolute()
    BIOFORMATS_MEMO_DIR.mkdir(exist_ok=True, parents=True)

# Java byte array size limit: 2^31 - 8 (leaves room for array header)
# Bio-Formats will fail with "Array size too large" if we exceed this.
# Key insight: This is a HARD limit in Java - can't be increased without JVM changes.
# Solution: Automatic tiling when reading >2GB planes (transparent to users)
MAX_JAVA_ARRAY_SIZE: int = 2**31 - 8
if _max_bytes := os.getenv("BIOFORMATS_MAX_JAVA_BYTES"):  # pragma: no cover
    try:
        MAX_JAVA_ARRAY_SIZE = int(_max_bytes)
    except ValueError:
        warnings.warn(
            f"Invalid BIOFORMATS_MAX_JAVA_BYTES: {_max_bytes!r}. "
            f"Using default {MAX_JAVA_ARRAY_SIZE}",
            stacklevel=2,
        )


class BioFile(Sequence[Series]):
    """Read image and metadata from file supported by Bioformats.

    BioFile instances must be explicitly opened before use, either by:

    1. Using a context manager: `with BioFile(path) as bf: ...`
    2. Explicitly calling `open()`: `bf = BioFile(path).open()`

    The recommended pattern is to use the context manager, which automatically
    handles opening and closing the file and cleanup of Java resources; but many usage
    patterns *will* also require explicit open/close.

    BioFile instances are not thread-safe. Create separate instances per thread.

    Lifecycle
    ---------
    BioFile manages the underlying Java reader through three states:

        UNINITIALIZED ── open() ──> OPEN ── close() ──> SUSPENDED
             ↑    ↑                  │ ↑                     │
             │    └── destroy() ─────┘ └──── open() ─────────┘
             └─────── destroy() ─────────────────────────────┘

    - `open()` first call: full initialization via `Java:setId()` (slow).
    - `close()`: releases file handles but preserves all parsed metadata
      and reader state in memory (`Java:close(fileOnly=true)`).
    - `open()` after `close()`: fast: simply reacquires the file handle — no re-parsing.
    - `destroy()` / `__exit__()`: full teardown — releases the Java
      reader and all cached state, returning to `UNINITIALIZED`.  `open()`
      can be called again but will require full re-initialization.
    - `__del__` GC finalizer: equivalent to `destroy()`.

    `SUSPENDED` preserves the Java reader's parsed state (format-specific
    headers, CoreMetadata, OME-XML DOM, metadata hashtable) in JVM heap,
    as well as Python-side `core_metadata()`. Only file handles are released.
    This is what enables the fast `open()` path.

    `destroy()` releases all of these, making the Java objects eligible for
    JVM garbage collection. The `BioFile` reverts to its initial state.

    Parameters
    ----------
    path : str or Path
        Path to file
    meta : bool, optional
        Whether to get metadata as well, by default True
    group_files : bool or None, optional
        Whether SCIFIO should group related files in a multi-file dataset (e.g. a
        directory of single-plane TIFFs) into one logical image. Maps to SCIFIO's
        ``SCIFIOConfig.groupableSetGroupFiles``. If `None` (default), the format's
        own default behavior is used.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        meta: bool = True,
        group_files: bool | None = None,
    ):
        self._path = str(path)
        self._lock = RLock()
        self._meta = meta
        self._group_files = group_files

        # Reader and finalizer created in open()
        self._java_reader: IFormatReader | None = None
        # SCIFIO Metadata (io.scif.Metadata) for the open reader
        self._java_metadata: Any = None
        # 2D structure: list[series][resolution]
        self._core_meta_list: list[list[CoreMetadata]] | None = None
        # maps (series, resolution) -> flat SCIFIO image index used by openPlane
        self._image_index: list[list[int]] | None = None
        self._cached_ome_meta: OME | None = None
        self._cached_ome_xml: str | None = None
        self._finalizer: weakref.finalize | None = None
        # _suspended is the user-facing logical state toggled by open()/close().
        # _source_live tracks whether the Java source handle is actually acquired —
        # reads transparently re-acquire it without changing the logical state, which
        # mirrors Bio-Formats' on-demand reopen behavior.
        self._suspended: bool = False
        self._source_live: bool = False

    def core_metadata(self, series: int = 0, resolution: int = 0) -> CoreMetadata:
        """Get metadata for specified series and resolution.

        Parameters
        ----------
        series : int, optional
            Series index, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.  Negative indexing
            supported (e.g., -1 = lowest resolution).

        Returns
        -------
        CoreMetadata
            Metadata for the specified series and resolution

        Raises
        ------
        RuntimeError
            If file is not open
        IndexError
            If series or resolution index is out of bounds

        Notes
        -----
        Resolution support is included for future compatibility, but currently
        only resolution 0 (full resolution) is exposed in the public API.
        """
        if self._core_meta_list is None:
            raise RuntimeError("File not open - call open() first")
        if series < 0 or series >= len(self._core_meta_list):
            raise IndexError(
                f"Series index {series} out of range "
                f"(file has {len(self._core_meta_list)} series)"
            )
        n_res = len(self._core_meta_list[series])
        resolution = _normalize_resolution(resolution, n_res)
        return self._core_meta_list[series][resolution]

    def global_metadata(self) -> Mapping[str, Any]:
        """Return the global metadata as a dictionary.

        This includes all metadata that is not specific to a particular series or
        resolution. The exact contents will depend on the file and reader, but may
        include things like instrument information, acquisition settings, and other
        annotations.

        Returns
        -------
        Mapping[str, Any]
            Global metadata key-value pairs

        Raises
        ------
        RuntimeError
            If file is not open
        """
        self._ensure_java_reader()
        table = self._java_metadata.getTable()
        if table is None:  # pragma: no cover
            return {}
        return {str(k): jtype_to_python(v) for k, v in table.items()}

    def open(self) -> Self:
        """Open file and initialize reader, or re-open if previously closed.

        On first call, performs full initialization (`setId`). If the file
        was previously closed via `close()`, reopens cheaply by reacquiring
        only the file handle without re-parsing.

        Safe to call multiple times — no-op if already open.

        Returns
        -------
        Self
            Returns `self` for method chaining.

        Examples
        --------
        ```python
        # Method chaining
        bf = BioFile(path).open()

        # Or two lines
        bf = BioFile(path)
        bf.open()
        ```

        See Also
        --------
        ensure_open : Context manager that restores previous state on exit
        """
        with self._lock:
            if self._java_reader is not None and not self._suspended:
                return self  # Already open

            if self._suspended:
                # Fast path: reacquire the file handle (if a read hasn't already done
                # so transparently) and clear the logical suspended state. Parsed
                # Python-side metadata is preserved.
                try:
                    self._acquire_source()
                except Exception:
                    self.destroy()
                    raise
                self._suspended = False
                return self

            # Full initialization (first open, or after context-manager exit).
            # SCIFIO's initializer auto-detects the format and returns a Reader.
            scifio = get_scifio()
            r = scifio.initializer().initializeReader(
                self._make_location(), self._make_config()
            )
            try:
                core_meta, image_index = self._get_core_metadata(r)
            except Exception:  # pragma: no cover
                with suppress(Exception):
                    r.close()
                raise

            self._java_reader = r
            self._java_metadata = r.getMetadata()
            self._core_meta_list = core_meta
            self._image_index = image_index
            self._source_live = True
            self._finalizer = weakref.finalize(self, _close_java_reader, r)
        return self

    def _acquire_source(self) -> None:
        """Re-acquire the file handle for a suspended reader.

        Unlike Bio-Formats, a SCIFIO reader cannot reopen after ``close(fileOnly)``
        (both ``setSource`` and ``openPlane`` raise). So we fully release the stale
        reader and initialize a fresh one for the same file. The cached Python-side
        metadata (``_core_meta_list`` / ``_image_index``) is unchanged because the file
        is identical.
        """
        if self._source_live:
            return
        # Drop the stale reader (and its finalizer) and build a fresh one.
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        scifio = get_scifio()
        r = scifio.initializer().initializeReader(
            self._make_location(), self._make_config()
        )
        self._java_reader = r
        self._java_metadata = r.getMetadata()
        self._source_live = True
        self._finalizer = weakref.finalize(self, _close_java_reader, r)

    def _make_location(self) -> Any:
        """Wrap the file path in a SciJava ``FileLocation`` for SCIFIO."""
        FileLocation = jimport("org.scijava.io.location.FileLocation")
        return FileLocation(os.path.abspath(self._path))

    def _make_config(self) -> Any:
        """Build the ``SCIFIOConfig`` for reader initialization.

        A config must always be supplied: the no-config code path uses name-only
        format detection, which fails for formats that require inspecting file
        content (e.g. CZI).
        """
        SCIFIOConfig = jimport("io.scif.config.SCIFIOConfig")
        config = SCIFIOConfig()
        if self._group_files is not None:
            config.groupableSetGroupFiles(self._group_files)
        return config

    def ensure_open(self) -> _EnsureOpenContext:
        """Context manager that temporarily opens the file if closed.

        Opens the file if needed, then on exit: suspends if it started closed
        (uninitialized or suspended), or leaves open if it started open. This
        allows temporary access without disrupting the caller's file state.

        Note: "closed" encompasses both uninitialized and suspended states.
        Files starting uninitialized will end suspended (not destroyed).

        Returns
        -------
        _EnsureOpenContext
            Context manager that restores open/closed state on exit.

        Examples
        --------
        ```python
        bf = BioFile(path)

        # Started uninitialized -> ends suspended
        with bf.ensure_open():
            data = bf.read_plane()
        assert bf.suspended  # not destroyed

        # Started open -> stays open
        bf.open()
        with bf.ensure_open():
            data = bf.read_plane()
        assert not bf.closed
        ```
        """
        return _EnsureOpenContext(self, close_on_exit=self.closed)

    def close(self) -> None:
        """Close file handles while preserving reader state for fast reopen.

        Releases the underlying file handle via Java `close(fileOnly=true)`
        but keeps all parsed metadata and reader state in memory. A subsequent
        `open()` call will cheaply reacquire the file handle.

        Metadata remains accessible via `core_metadata()` and `len()`
        while the file is closed.

        Safe to call multiple times — no-op if already closed.
        """
        with self._lock:
            if self._java_reader is not None:
                if self._source_live:
                    self._java_reader.close(True)  # fileOnly=True
                    self._source_live = False
                self._suspended = True

    def destroy(self) -> None:
        """Full cleanup — release all Java resources and cached state.

        Releases the Java reader and all parsed metadata, returning to
        UNINITIALIZED. `open()` can be called again but will require full
        re-initialization (slow path). The GC finalizer calls this
        automatically if not called explicitly.

        Safe to call multiple times or from any state.
        """
        with self._lock:
            if self._finalizer is not None:
                self._finalizer()
                self._finalizer = None
            self._java_reader = None
            self._java_metadata = None
            self._core_meta_list = None
            self._image_index = None
            self._suspended = False
            self._source_live = False

    def as_array(self, series: int = 0, resolution: int = 0) -> LazyBioArray:
        """Return a lazy numpy-compatible array that reads data on-demand.

        The returned array behaves like a numpy array but reads data from disk
        only when indexed. Use it just like a numpy array - any indexing operation
        will read only the requested planes or sub-regions, not the entire dataset.

        Supports integer and slice indexing on all dimensions. The array also
        implements the `__array__()` protocol for seamless numpy integration.

        Parameters
        ----------
        series : int, optional
            Series index, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.  Negative indexing
            supported (e.g., -1 = lowest resolution).

        Returns
        -------
        LazyBioArray
            Lazy array in (T, C, Z, Y, X) or (T, C, Z, Y, X, rgb) format

        Examples
        --------
        Index like a numpy array - only reads what you request:

        >>> with BioFile("image.nd2") as bf:
        ...     arr = bf.as_array()  # No data read yet
        ...
        ...     # Read single plane (t=0, c=0, z=2)
        ...     plane = arr[0, 0, 2]  # Only this plane read from disk
        ...
        ...     # Read all timepoints for one channel/z
        ...     timeseries = arr[:, 0, 2]  # Reads T planes
        ...
        ...     # Read sub-region across entire volume
        ...     roi = arr[:, :, :, 100:200, 50:150]  # Reads 100x50 sub-regions
        ...
        ...     # Materialize entire dataset (two equivalent ways)
        ...     full_data = arr[:]  # Using slice notation
        ...     full_data = np.array(arr)  # Using numpy conversion

        Notes
        -----
        BioFile must remain open while using the array. Multiple arrays can
        coexist, each reading from their own series independently.

        Planes >2GB automatically use tiled reading (transparent, ~20% slower).
        """
        from scyfio._lazy_array import LazyBioArray

        meta0 = self.core_metadata(series)  # validates series
        resolution = _normalize_resolution(resolution, meta0.resolution_count)
        return LazyBioArray(self, series, resolution)

    @overload
    def to_zarr_store(
        self,
        series: Literal[None] = ...,
        *,
        tile_size: tuple[int, int] | None = ...,
    ) -> BFOmeZarrStore: ...
    @overload
    def to_zarr_store(
        self,
        series: int,
        resolution: int = ...,
        *,
        tile_size: tuple[int, int] | None = ...,
    ) -> BFArrayStore: ...
    def to_zarr_store(
        self,
        series: int | None = None,
        resolution: int = 0,
        *,
        tile_size: tuple[int, int] | None = None,
    ) -> BFOmeZarrStore | BFArrayStore:
        """Return a zarr v3 group store containing all series and resolutions.

        Creates an OME-ZARR group structure following NGFF v0.5 specification,
        with full hierarchy including all series and resolution levels. Useful
        for tools like napari that expect complete OME-ZARR groups.

        Directory structure::

            root/
            ├── zarr.json (group metadata)
            ├── OME/
            │   ├── zarr.json (series list)
            │   └── METADATA.ome.xml (raw OME-XML)
            ├── 0/ (series 0 - multiscales group)
            │   ├── zarr.json (multiscales metadata with axes/datasets)
            │   ├── 0/ (full resolution)
            │   │   ├── zarr.json (array metadata)
            │   │   └── c/... (chunk data)
            │   └── 1/ (downsampled, if exists)
            └── 1/ (series 1, if exists)

        Parameters
        ----------
        series : int, optional
            If provided, only return the specified series and resolution as a single
            array store. By default, returns the full group store with all series and
            resolutions.
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.  Negative indexing
            supported (e.g., -1 = lowest resolution).
        tile_size : tuple[int, int], optional
            If provided, Y and X are chunked into tiles of this size instead of
            full planes. Chunk shape becomes ``(1, 1, 1, tile_y, tile_x)``.

        Returns
        -------
        BFOmeZarrStore
            Read-only zarr v3 Store containing the full file hierarchy.

        Examples
        --------
        Open as zarr group and access arrays:

        >>> import zarr
        >>> with BioFile("image.nd2") as bf:
        ...     group = zarr.open_group(bf.to_zarr_store(), mode="r")
        ...     # Access first series, full resolution
        ...     arr = group["0/0"]
        ...     data = arr[0, 0, 0]
        ...
        ...     # Check multiscales metadata
        ...     print(group["0"].attrs["ome"]["multiscales"])
        ...
        ...     # Save to disk
        ...     bf.to_zarr_store().save("output.ome.zarr")

        Notes
        -----
        - For single array access, prefer `as_array().to_zarr_store()` (simpler)
        - This creates the full hierarchy needed for multi-series/multi-resolution
          visualization tools
        - Conforms to NGFF v0.5 specification
        """
        if series is None:
            from scyfio._zarr._group_store import BFOmeZarrStore

            return BFOmeZarrStore(self, tile_size=tile_size)

        lazy = self.as_array(series=series, resolution=resolution)
        return lazy.to_zarr_store(tile_size=tile_size)

    def to_dask(
        self,
        series: int = 0,
        resolution: int = 0,
        *,
        chunks: str | tuple = "auto",
        tile_size: tuple[int, int] | str | None = None,
    ) -> dask.array.Array:
        """Create dask array for lazy computation on Bio-Formats data.

        Returns a dask array in TCZYX[r] order that wraps a
        [`LazyBioArray`][scyfio.LazyBioArray]. Uses single-threaded scheduler
        for Bio-Formats thread safety.

        Parameters
        ----------
        series : int, optional
            Series index to read from, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.  Negative indexing
            supported (e.g., -1 = lowest resolution).
        chunks : str or tuple, default "auto"
            Chunk specification. Examples:
            - "auto": Let dask decide (default)
            - (1, 1, 1, -1, -1): Full Y,X planes per T,C,Z
            - (1, 1, 1, 512, 512): 512x512 tiles
            Mutually exclusive with tile_size.
        tile_size : tuple[int, int] or "auto", optional
            Tile-based chunking for Y,X dimensions (T,C,Z get chunks of 1).
            - (512, 512): Use 512x512 tiles
            - "auto": Query Bio-Formats optimal tile size
            Mutually exclusive with chunks.

        Returns
        -------
        dask.array.Array
            Dask array that reads data on-demand. Shape is (T, C, Z, Y, X) or
            (T, C, Z, Y, X, rgb) for RGB images.

        Raises
        ------
        ValueError
            If both chunks and tile_size are specified

        Examples
        --------
        >>> with BioFile("image.nd2") as bf:
        ...     darr = bf.to_dask(chunks=(1, 1, 1, -1, -1))
        ...     result = darr.mean(axis=2).compute()  # Z-projection

        Notes
        -----
        - BioFile must remain open during computation
        - Uses synchronous scheduler by default (required for thread safety)
        """
        lazy_arr = self.as_array(series=series, resolution=resolution)
        return lazy_arr.to_dask(chunks=chunks, tile_size=tile_size)

    def to_xarray(self, series: int = 0, resolution: int = 0) -> xr.DataArray:
        """Return xarray.DataArray for specified series and resolution.

        The returned DataArray has `.dims` and `.coords` attributes populated according
        to the metadata. Dimension and coord names will be one of: `TCZYXS`, where `S`
        represents the RGB/RGBA channels if present. The parsed `ome_types.OME` object
        is also available in the `.attrs['ome_metadata']` attribute of the DataArray.

        Parameters
        ----------
        series : int, optional
            Series index to read from, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.  Negative indexing
            supported (e.g., -1 = lowest resolution).
        """
        return self.as_array(series=series, resolution=resolution).to_xarray()

    @property
    def closed(self) -> bool:
        """Return True if the file is currently closed (uninitialized or suspended)."""
        return self._java_reader is None or self._suspended

    @property
    def suspended(self) -> bool:
        """Return True if the file is currently suspended (closed but not destroyed).

        "Suspended" means:
        - we have previously opened the file and parsed the metadata
        - we retain the initialized Java reader and its state in the JVM
        - but we have released the file handles to free up system resources
        """
        return self._java_reader is not None and self._suspended

    @property
    def filename(self) -> str:
        """Return name of file handle."""
        return self._path

    @property
    def ome_xml(self) -> str:
        """Return plain OME XML string.

        SCIFIO produces OME-XML by *translation*: the format-specific metadata is
        translated into an ``io.scif.ome.OMEMetadata`` object, whose root is a
        Bio-Formats ``OMEXMLMetadata`` that can dump XML.
        """
        self._ensure_java_reader()  # validate open state
        if not self._meta:
            return ""
        if self._cached_ome_xml is None:
            self._cached_ome_xml = ""
            try:
                scifio = get_scifio()
                OMEMetadata = jimport("io.scif.ome.OMEMetadata")
                omexml = OMEMetadata(scifio.getContext())
                scifio.translator().translate(self._java_metadata, omexml, True)
                self._cached_ome_xml = str(omexml.getRoot().dumpXML())
            except Exception as e:
                warnings.warn(
                    f"Failed to retrieve OME XML: {e}", RuntimeWarning, stacklevel=2
                )
        return self._cached_ome_xml

    @property
    def ome_metadata(self) -> OME:
        """Return [`ome_types.OME`][] object parsed from OME XML."""
        if self._cached_ome_meta is None:
            if not (omx_xml := self.ome_xml):  # pragma: no cover (not sure if possible)
                self._cached_ome_meta = OME()
            else:
                xml = _utils.clean_ome_xml_for_known_issues(omx_xml)
                self._cached_ome_meta = OME.from_xml(xml)
        return self._cached_ome_meta

    def __enter__(self) -> Self:
        """Enter context manager - ensures file is open."""
        self.open()  # Idempotent, so safe to call
        return self

    def __exit__(self, *_args: Any) -> None:
        """Exit context manager — full resource cleanup, including the Java reader."""
        self.destroy()

    def __len__(self) -> int:
        """Return the number of series in the file."""
        if self._core_meta_list is None:
            raise RuntimeError("File not open - call open() first")
        return len(self._core_meta_list)

    def series_count(self) -> int:
        """Return the number of series in the file (same as [`__len__`][scyfio.BioFile.__len__])."""  # noqa: E501
        return len(self)

    @overload
    def __getitem__(self, index: int) -> Series: ...
    @overload
    def __getitem__(self, index: slice) -> list[Series]: ...
    def __getitem__(self, index: int | slice) -> Series | list[Series]:
        """Return a [`Series`][scyfio.Series] proxy for the given index.

        Parameters
        ----------
        index : int | slice
            Series index or slice of series indices to retrieve
        """
        if isinstance(index, int):
            return self._get_series(index)
        elif isinstance(index, slice):
            return [self._get_series(i) for i in range(*index.indices(len(self)))]
        raise TypeError(f"Invalid index type: {type(index)}")  # pragma: no cover

    def used_files(self, *, metadata_only: bool = False) -> list[str]:
        """Return complete list of files needed to open this dataset.

        Parameters
        ----------
        metadata_only : bool, optional
            If True, only return files that do not contain pixel data (e.g., metadata,
            companion files, etc...), by default `False`. Only honored for formats read
            through the Bio-Formats compatibility layer.
        """
        self._ensure_java_reader()
        if (bf_reader := _bf_underlying_reader(self._java_metadata)) is not None:
            with suppress(Exception):
                return [str(x) for x in bf_reader.getUsedFiles(metadata_only) or ()]
        return [self._path]

    def lookup_table(self, series: int = 0) -> np.ndarray | None:
        """Return the color lookup table for an indexed-color series.

        Parameters
        ----------
        series : int, optional
            Series index, by default 0.

        Returns
        -------
        np.ndarray | None
            Array of shape `(N, 3)` with RGB values for each pixel index,
            or None if the series is not indexed.

        Examples
        --------

        To convert this object to a [`cmap.Colormap`][]:

        ``python
        import cmap
        from scyfio import BioFile

        with BioFile("indexed_image.ome.tiff") as bf:
            lut = bf.lookup_table()
            if lut is not None:
                colormap = cmap.Colormap(lut / lut.max())  # Normalize to [0, 1]
        ``
        """
        # `is_indexed` is a bit of a misnomer here (but bioformats uses it)
        # it really means "a LUT exists", not "the image is palette-based"
        #   - True palette images (GIF, indexed PNG) → is_indexed=True, is_rgb=False
        #   - Fluorescence with display LUT (ND2, CZI) → is_indexed=True, is_rgb=False
        #   - Plain grayscale (basic TIFF, no LUT) → is_indexed=False, is_rgb=False
        #   - RGB images → is_indexed=False, is_rgb=True
        if not self.core_metadata(series).is_indexed:
            return None

        self._ensure_java_reader()
        image_index = self._image_index[series][0]  # type: ignore[index]
        # SCIFIO exposes color tables on metadata that implements HasColorTable.
        with suppress(Exception):
            color_table = self._java_metadata.getColorTable(image_index, 0)
            if color_table is not None:
                return _color_table_to_numpy(color_table)
        return None

    def __iter__(self) -> Iterator[Series]:
        """Iterate over all series in the file."""
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        name = Path(self._path).name
        if self.closed:
            return f"BioFile('{name}', closed)"
        return f"BioFile('{name}', {len(self)} series)"

    def read_plane(
        self,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        y: slice | None = None,
        x: slice | None = None,
        series: int = 0,
        resolution: int = 0,
        buffer: np.ndarray | None = None,
    ) -> np.ndarray:
        """Read a single plane or sub-region directly from Bio-Formats.

        Low-level method wrapping Bio-Formats' `openBytes()` API. Provides
        fine-grained control for reading specific planes or rectangular
        sub-regions. Most users should use `as_array()` or `to_dask()` instead.

        **Not thread-safe.** Create separate BioFile instances per thread.

        Parameters
        ----------
        t : int, optional
            Time index, by default 0
        c : int, optional
            Channel index, by default 0
        z : int, optional
            Z-slice index, by default 0
        y : slice, optional
            Y-axis slice (default: full height). Example: `slice(100, 200)`
        x : slice, optional
            X-axis slice (default: full width). Example: `slice(50, 150)`
        series : int, optional
            Series index, by default 0
        resolution : int, optional
            Resolution level (0 = full resolution), by default 0.  Negative indexing
            supported (e.g., -1 = lowest resolution).
        buffer : np.ndarray, optional
            Pre-allocated buffer for efficient reuse in loops

        Returns
        -------
        np.ndarray
            Shape (height, width) for grayscale or (height, width, rgb) for RGB

        Examples
        --------
        >>> with BioFile("image.nd2") as bf:
        ...     plane = bf.read_plane(t=0, c=1, z=5)
        ...     roi = bf.read_plane(y=slice(200, 300), x=slice(200, 300))

        See Also
        --------
        as_array : Create a numpy-compatible lazy array
        to_dask : Create a dask array for lazy loading
        """
        reader = self._ensure_java_reader()
        n_res = len(self._core_meta_list[series])  # type: ignore[index]
        resolution = _normalize_resolution(resolution, n_res)
        image_index = self._image_index[series][resolution]  # type: ignore[index]

        # Get metadata for this series/resolution
        meta = self.core_metadata(series, resolution)
        shape = meta.shape

        # Validate plane coordinates (SCIFIO does not always raise on overflow).
        for name, value, limit in (
            ("t", t, shape.t),
            ("c", c, shape.c),
            ("z", z, shape.z),
        ):
            if not 0 <= value < limit:
                raise IndexError(
                    f"{name}={value} out of range for this image (0 to {limit - 1})"
                )

        # Handle default slices
        y = y if y is not None else slice(0, shape.y)
        x = x if x is not None else slice(0, shape.x)

        # Call optimized internal method
        im = self._read_plane(reader, meta, image_index, t, c, z, y, x)

        # If buffer provided, copy into it (for reuse in loops)
        if buffer is not None:
            buffer[:] = im
            return buffer

        return im

    # ========================== static methods ==========================

    @staticmethod
    def scifio_version() -> str:
        """Get the version of SCIFIO."""
        try:
            return str(get_scifio().getVersion())
        except Exception:  # pragma: no cover
            return "unknown"

    # Deprecated alias retained for backwards compatibility.
    @staticmethod
    def bioformats_version() -> str:
        """Deprecated alias for [`scifio_version`][scyfio.BioFile.scifio_version]."""
        return BioFile.scifio_version()

    @staticmethod
    def maven_coordinate() -> str:
        """Return the Maven coordinate used to load SCIFIO (scifio-bf-compat)."""
        from ._java_stuff import MAVEN_COORDINATE

        return MAVEN_COORDINATE

    # Deprecated alias retained for backwards compatibility.
    @staticmethod
    def bioformats_maven_coordinate() -> str:
        """Deprecated alias for [`maven_coordinate`][scyfio.BioFile.maven_coordinate]."""  # noqa: E501
        return BioFile.maven_coordinate()

    @staticmethod
    @cache
    def list_supported_suffixes() -> set[str]:
        """List all file suffixes supported by the available SCIFIO formats."""
        suffixes: set[str] = set()
        for fmt in get_scifio().format().getAllFormats():
            with suppress(Exception):
                suffixes.update(str(s) for s in fmt.getSuffixes())
        return suffixes

    @staticmethod
    @cache
    def list_available_readers() -> list[ReaderInfo]:
        """List all available SCIFIO formats.

        Returns
        -------
        list[ReaderInfo]
            Information about each available format, including:

            - format: human-readable format name (e.g., "Nikon ND2")
            - suffixes: supported file extensions (e.g., ("nd2", "jp2"))
            - class_name: full Java class name of the Format
            - is_gpl: best-effort license flag. SCIFIO does not expose per-format
              licensing, so this is always ``False`` (unknown).
        """
        formats = []
        for fmt in get_scifio().format().getAllFormats():
            class_name = str(fmt.getClass().getName()).removeprefix("io.scif.formats.")
            formats.append(
                ReaderInfo(
                    format=str(fmt.getFormatName()),
                    suffixes=tuple(str(s) for s in fmt.getSuffixes()),
                    class_name=class_name,
                    is_gpl=False,
                )
            )
        return formats

    # ========================== Internal methods ==========================

    def _ensure_java_reader(self) -> IFormatReader:
        """Return the native reader, raising if never opened.

        If the reader is suspended (file handle released via ``close()``) it is
        transparently resumed first — SCIFIO readers, unlike Bio-Formats, do not
        re-acquire the source on demand, so we do it here. This keeps lazy consumers
        (e.g. zarr stores that outlive an ``ensure_open()`` block) working.
        """
        if self._java_reader is None:
            raise RuntimeError("File not open - call open() first")
        # Transparently re-acquire the source handle if it was released by close(),
        # without clearing the logical suspended state (mirrors Bio-Formats).
        self._acquire_source()
        return self._java_reader

    def _get_core_metadata(
        self, reader: IFormatReader
    ) -> tuple[list[list[CoreMetadata]], list[list[int]]]:
        """Parse SCIFIO metadata into the 2D ``[series][resolution]`` structure.

        Returns both the metadata grid and a parallel grid mapping each
        ``(series, resolution)`` to the flat SCIFIO image index used by
        ``reader.openPlane``.

        SCIFIO has no native concept of resolution levels — each "image" is a single
        plane stack. The Bio-Formats compatibility layer flattens Bio-Formats
        resolution levels into separate SCIFIO images. To reconstruct pyramids we peek
        at the underlying Bio-Formats reader's ``CoreMetadataList`` (which still carries
        the per-series ``resolutionCount`` even when flattened) and chunk the flat image
        list accordingly. Native SCIFIO formats fall through to one resolution per
        image.
        """
        meta = reader.getMetadata()
        image_count = int(reader.getImageCount())

        # Resolution grouping: how many consecutive images form each series.
        # Only the Bio-Formats compatibility layer exposes pyramids; everything else
        # gets one resolution per image.
        res_counts = _bf_resolution_counts(meta, image_count)

        # Per-image CoreMetadata, derived from the SCIFIO ImageMetadata so that plane
        # indexing (which goes through SCIFIO's axis model) stays consistent.
        per_image = [
            CoreMetadata.from_image_metadata(meta.get(i)) for i in range(image_count)
        ]

        result: list[list[CoreMetadata]] = []
        image_index: list[list[int]] = []
        flat = 0
        for count in res_counts:
            group = per_image[flat : flat + count]
            for cm in group:
                cm.resolution_count = count
            result.append(group)
            image_index.append(list(range(flat, flat + count)))
            flat += count
        return result, image_index

    def get_thumbnail(
        self,
        series: int = 0,
        *,
        t: int = 0,
        c: int = 0,
        z: int | None = None,
        max_thumbnail_size: int | tuple[int, int] = 128,
        max_read_size: int = 4096,
    ) -> np.ndarray:
        """Get thumbnail image for specified series.

        Returns a downsampled version of the specified plane from the specified series,
        channel, timepoint, and z-slice (default: central slice). The thumbnail is
        scaled to fit within `max_thumbnail_size` pixels while maintaining aspect ratio.

        This method reads the _lowest_ resolution level available for the series, and
        then downsamples it to create the thumbnail. If the lowest resolution is still
        larger than `max_read_size`, it will read a centered sub-region of the lowest
        resolution, no greater than `max_read_size` in either dimension.

        !!! note
            For stability and performance, this does *not* use the java openThumbBytes
            API.

        Parameters
        ----------
        series : int, optional
            Series index to get thumbnail from, by default 0
        t : int, optional
            Time index for thumbnail plane, by default 0
        c : int, optional
            Channel index for thumbnail plane, by default 0
        z : int | None, optional
            Z-slice index for thumbnail plane, by default None (take the central slice)
        max_thumbnail_size : int | tuple[int, int], optional
            Maximum thumbnail size. If int, limits both width and height to this value.
            If tuple, interpreted as ``(max_width, max_height)``.
        max_read_size : int, optional
            Maximum dimension size to read directly from Bio-Formats before switching.
            If this is lower than the size of the full plane, the image will be cropped.
            Decrease for speed, increase for field of view.

        Returns
        -------
        np.ndarray
            Thumbnail image as numpy array with shape (H, W) for grayscale or
            (H, W, RGB) for RGB images.
        """
        reader = self._ensure_java_reader()

        # Get lowest resolution of requested series + downscale
        low_res = self.core_metadata(series=series).resolution_count - 1
        low_meta = self.core_metadata(series=series, resolution=low_res)
        sy, sx = low_meta.shape.y, low_meta.shape.x

        # Cap read size if lowest resolution is still large
        if sy > max_read_size or sx > max_read_size:
            scale = max(sy / max_read_size, sx / max_read_size)
            read_h = max(1, round(sy / scale))
            read_w = max(1, round(sx / scale))
            y_start = (sy - read_h) // 2
            x_start = (sx - read_w) // 2
        else:
            read_h, read_w = sy, sx
            y_start, x_start = 0, 0

        tz = low_meta.shape.z // 2 if z is None else z
        image_index = self._image_index[series][low_res]  # type: ignore[index]
        with self._lock:
            img = self._read_plane_direct(
                reader,
                low_meta,
                image_index,
                t,
                c,
                tz,
                y_start,
                x_start,
                read_h,
                read_w,
            )

        target_x, target_y = _thumbnail_target_size(sx, sy, max_size=max_thumbnail_size)
        return _resize_thumbnail(img, target_h=target_y, target_w=target_x)

    def _get_series(self, index: int) -> Series:
        """Internal method to get a Series with index validation."""
        n = len(self)  # also validates open state
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"Series index {index} out of range (file has {n} series)")
        from scyfio._series import Series

        return Series(self, index)

    def _plane_index(self, image_index: int, z: int, c: int, t: int) -> int:
        """Map a ``(z, c, t)`` position to a SCIFIO plane (raster) index.

        SCIFIO's plane index is a raster over the image's *non-planar* axes. We walk
        those axes, placing ``z`` on the Z axis, ``t`` on the Time axis, and
        distributing the channel index ``c`` across any remaining (channel-like) axes
        — Bio-Formats RGB data can introduce extra channel axes with non-standard
        labels. Verified pixel-identical to Bio-Formats ``getIndex(z, c, t)``.
        """
        Axes = jimport("net.imagej.axis.Axes")
        imeta = self._java_metadata.get(image_index)
        planar = int(imeta.getPlanarAxisCount())
        n = int(imeta.getAxes().size())

        lengths: list[int] = []
        pos: list[int] = []
        chan_dims: list[int] = []
        for d in range(planar, n):
            axis_type = imeta.getAxis(d).type()
            lengths.append(int(imeta.getAxisLength(axis_type)))
            if axis_type == Axes.Z:
                pos.append(z)
            elif axis_type == Axes.TIME:
                pos.append(t)
            else:
                pos.append(0)
                chan_dims.append(len(lengths) - 1)

        if not lengths:
            return 0

        # spread the single channel index across channel-like axes (first fastest)
        remaining = c
        for d in chan_dims:
            pos[d] = remaining % lengths[d]
            remaining //= lengths[d]

        FormatTools = jimport("io.scif.util.FormatTools")
        la = jpype.JArray(jpype.JLong)(lengths)
        pa = jpype.JArray(jpype.JLong)(pos)
        return int(FormatTools.positionToRaster(la, pa))

    def _make_bounds(
        self, image_index: int, y_start: int, x_start: int, height: int, width: int
    ) -> Any:
        """Build an ``Interval`` over all planar axes for a sub-region read.

        The interval must span every planar axis (X, Y, and any colour-sample axis
        such as interleaved RGB), not just X/Y, or SCIFIO raises an index error.
        """
        Axes = jimport("net.imagej.axis.Axes")
        FinalInterval = jimport("net.imglib2.FinalInterval")
        imeta = self._java_metadata.get(image_index)
        planar = int(imeta.getPlanarAxisCount())
        mins: list[int] = []
        sizes: list[int] = []
        for d in range(planar):
            axis_type = imeta.getAxis(d).type()
            if axis_type == Axes.X:
                mins.append(x_start)
                sizes.append(width)
            elif axis_type == Axes.Y:
                mins.append(y_start)
                sizes.append(height)
            else:
                mins.append(0)
                sizes.append(int(imeta.getAxisLength(axis_type)))
        return FinalInterval.createMinSize(*(mins + sizes))

    def _read_plane(
        self,
        reader: IFormatReader,
        meta: CoreMetadata,
        image_index: int,
        t: int,
        c: int,
        z: int,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        """Fast plane reading for hot path (internal use only).


        This method skips all validation and metadata lookups, assuming they've been
        done once before entering a tight loop.

        It *does*, however, dispatch to tiled or direct read based on plane size.
        (Note: users have full power to control tiling via slicing into LazyBioArray,
        or by using to_dask()... this is just a safety net for requests that would
        exceed Java limits.)
        """
        shape = meta.shape
        y_start, y_stop, _ = y.indices(shape.y)
        x_start, x_stop, _ = x.indices(shape.x)

        height = y_stop - y_start
        width = x_stop - x_start
        plane_bytes = height * width * meta.dtype.itemsize * meta.shape.rgb

        if plane_bytes > MAX_JAVA_ARRAY_SIZE:
            return self._read_plane_tiled(
                reader, meta, image_index, t, c, z, y_start, x_start, height, width
            )
        return self._read_plane_direct(
            reader, meta, image_index, t, c, z, y_start, x_start, height, width
        )

    def _read_plane_direct(
        self,
        reader: IFormatReader,
        meta: CoreMetadata,
        image_index: int,
        t: int,
        c: int,
        z: int,
        y_start: int,
        x_start: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Read plane directly (fast path for <2GB planes)."""
        n_rgb = meta.shape.rgb
        dtype = meta.dtype
        plane_idx = self._plane_index(image_index, z, c, t)
        bounds = self._make_bounds(image_index, y_start, x_start, height, width)
        plane = reader.openPlane(image_index, plane_idx, bounds)
        im = np.frombuffer(memoryview(plane.getBytes()), dtype)  # type: ignore
        return _reshape_image_buffer(
            im,
            dtype=dtype,
            height=height,
            width=width,
            rgb=n_rgb,
            interleaved=meta.is_interleaved,
        )

    def _calculate_tile_height(self, meta: CoreMetadata, region_width: int) -> int:
        """Calculate max rows per tile respecting Java array limit and heap space.

        Uses full-width rows (no X tiling) to minimize openBytes() calls.
        """
        row_bytes = region_width * meta.dtype.itemsize * meta.shape.rgb

        # Constraint 1: Java's max byte array size
        tile_height = MAX_JAVA_ARRAY_SIZE // row_bytes

        # Constraint 2: Available heap (with 80% safety margin)
        rt = jimport("java.lang.Runtime").getRuntime()
        available_heap = rt.maxMemory() - (rt.totalMemory() - rt.freeMemory())
        max_heap_rows = int(available_heap * 0.8) // row_bytes

        return max(1, min(tile_height, max_heap_rows))

    def _read_plane_tiled(
        self,
        reader: IFormatReader,
        meta: CoreMetadata,
        image_index: int,
        t: int,
        c: int,
        z: int,
        y_start: int,
        x_start: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Read large plane via tiling to avoid 2GB Java array limit.

        Strategy: read full-width row-bands via ``openPlane`` sub-regions (each band's
        byte array stays under the Java limit) and copy them into the output array.
        """
        n_rgb = meta.shape.rgb
        dtype = meta.dtype

        # Preallocate output
        output_shape = (height, width, n_rgb) if n_rgb > 1 else (height, width)
        output = np.empty(output_shape, dtype=dtype)

        # Calculate tile size
        tile_height = self._calculate_tile_height(meta, width)

        plane_idx = self._plane_index(image_index, z, c, t)

        # Read tiles
        y_offset = 0
        for y0 in range(0, height, tile_height):
            h = min(tile_height, height - y0)

            bounds = self._make_bounds(image_index, y_start + y0, x_start, h, width)
            plane = reader.openPlane(image_index, plane_idx, bounds)

            # Copy tile data (count is elements, not bytes)
            tile_data = np.frombuffer(
                memoryview(plane.getBytes()),  # pyright: ignore[reportArgumentType]
                dtype=dtype,
                count=h * width * n_rgb,
            ).copy()

            # Copy to output
            if n_rgb > 1:
                if meta.is_interleaved:
                    output[y_offset : y_offset + h].ravel()[:] = tile_data
                else:
                    # Non-interleaved needs transpose
                    tile = tile_data.reshape(n_rgb, h, width).transpose(1, 2, 0)
                    output[y_offset : y_offset + h] = tile
            else:
                output[y_offset : y_offset + h].ravel()[:] = tile_data

            y_offset += h

        return output


def _bf_underlying_reader(java_metadata: Any) -> Any | None:
    """Return the wrapped Bio-Formats ``IFormatReader``, if any.

    Encapsulates all peeking into the Bio-Formats compatibility layer: only the
    ``io.scif.bf.BioFormatsFormat$Metadata`` class exposes ``getReader()``. Native
    SCIFIO formats return ``None``.
    """
    if java_metadata is None:  # pragma: no cover
        return None
    if "BioFormatsFormat" not in str(java_metadata.getClass().getName()):
        return None
    with suppress(Exception):
        return java_metadata.getReader()
    return None  # pragma: no cover


def _bf_resolution_counts(java_metadata: Any, image_count: int) -> list[int]:
    """Group flat SCIFIO images into per-series resolution-level counts.

    For Bio-Formats-backed metadata we peek the underlying reader's core-metadata
    list, whose per-series first entry carries ``resolutionCount`` even when
    resolutions are flattened into separate images. Anything else (native SCIFIO
    formats) gets one resolution per image.
    """
    bf_reader = _bf_underlying_reader(java_metadata)
    if bf_reader is not None:
        with suppress(Exception):
            core_list = bf_reader.getCoreMetadataList()
            counts: list[int] = []
            i = 0
            n = int(core_list.size())
            while i < n:
                rc = max(1, int(core_list.get(i).resolutionCount))
                counts.append(rc)
                i += rc
            if sum(counts) == image_count:
                return counts
    return [1] * image_count


def _color_table_to_numpy(color_table: Any) -> np.ndarray:
    """Convert an imglib2 ``ColorTable`` into an ``(N, components)`` numpy array."""
    components = int(color_table.getComponentCount())
    length = int(color_table.getLength())
    # ColorTable8 values are 0-255; ColorTable16 are 0-65535. Use a width that fits.
    dtype = np.uint8 if int(color_table.getBits()) <= 8 else np.uint16
    out = np.empty((length, components), dtype=dtype)
    for comp in range(components):
        for i in range(length):
            out[i, comp] = int(color_table.get(comp, i))
    return out


def _close_java_reader(java_reader: IFormatReader | None) -> None:
    """Close a Java reader if JVM is still running.

    Used as weakref finalizer for last-resort cleanup. This can ONLY close
    the Java file handle - it cannot access Python instance state because
    it's called after the BioFile instance is garbage collected.

    For explicit cleanup, use the BioFile.close() method instead.
    """
    # Only attempt close during normal operation (not shutdown)
    if java_reader is None or sys.is_finalizing() or not jpype.isJVMStarted():
        return  # pragma: no cover
    with suppress(Exception):
        java_reader.close()


class _EnsureOpenContext(AbstractContextManager[BioFile]):
    """A context manager that ensures BioFile is open and restores state on exit.

    Unlike BioFile.__enter__/__exit__ which destroys on exit, this context manager
    ensures the file is open for the duration of the block, then restores it to
    whatever state it was in before (open or closed).
    """

    def __init__(self, biofile: BioFile, close_on_exit: bool) -> None:
        self.biofile = biofile
        self.close_on_exit = close_on_exit

    def __enter__(self) -> BioFile:
        if self.biofile.closed:
            self.biofile.open()
        return self.biofile

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> Any:
        if self.close_on_exit:
            self.biofile.close()


def _reshape_image_buffer(
    data: np.ndarray,
    *,
    dtype: np.dtype[Any],
    height: int,
    width: int,
    rgb: int,
    interleaved: bool,
) -> np.ndarray:
    """Reshape flat pixel buffer into 2D/3D image array."""
    arr = np.asarray(data, dtype=dtype, copy=False)
    if rgb > 1:
        if interleaved:
            return arr.reshape(height, width, rgb)
        return arr.reshape(rgb, height, width).transpose(1, 2, 0)
    return arr.reshape(height, width)


def _normalize_resolution(resolution: int, resolution_count: int) -> int:
    if resolution < 0:
        resolution += resolution_count
    if not 0 <= resolution < resolution_count:
        raise IndexError(
            f"Resolution {resolution} out of range (0 to {resolution_count - 1})"
        )
    return resolution


def _normalize_thumbnail_max_size(max_size: int | tuple[int, int]) -> tuple[int, int]:
    """Return `(max_width, max_height)` with validation."""
    if isinstance(max_size, int):
        max_size = (max_size, max_size)
    if len(max_size) != 2:  # pragma: no cover
        raise ValueError("max_size must be an int or a tuple of two ints")
    max_width, max_height = max_size
    if max_width < 1 or max_height < 1:  # pragma: no cover
        raise ValueError("max_size values must be >= 1")
    return max_width, max_height


def _thumbnail_target_size(
    width: int, height: int, *, max_size: int | tuple[int, int]
) -> tuple[int, int]:
    """Fit source size inside a box, preserving aspect ratio and never upscaling."""
    max_width, max_height = _normalize_thumbnail_max_size(max_size)
    if width <= max_width and height <= max_height:
        return max(1, width), max(1, height)

    scale = min(max_width / width, max_height / height)
    target_width = max(1, min(max_width, round(width * scale)))
    target_height = max(1, min(max_height, round(height * scale)))
    return target_width, target_height


def _resize_thumbnail(img: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
    """Resize image to exact thumbnail shape using pure NumPy.

    Uses area averaging for downscaling and nearest-neighbor for upscaling.
    """
    h, w = img.shape[:2]
    if (h, w) == (target_h, target_w):  # pragma: no cover
        return img

    if target_h >= h or target_w >= w:
        y_idx = np.linspace(0, h - 1, target_h, dtype=np.intp)
        x_idx = np.linspace(0, w - 1, target_w, dtype=np.intp)
        return img[y_idx][:, x_idx]

    y_edges = np.linspace(0, h, target_h + 1)
    x_edges = np.linspace(0, w, target_w + 1)

    y0 = np.floor(y_edges[:-1]).astype(np.intp)
    y1 = np.ceil(y_edges[1:]).astype(np.intp)
    x0 = np.floor(x_edges[:-1]).astype(np.intp)
    x1 = np.ceil(x_edges[1:]).astype(np.intp)

    y1 = np.maximum(y1, y0 + 1)
    x1 = np.maximum(x1, x0 + 1)

    out_shape = (target_h, target_w, *img.shape[2:])
    out = np.empty(out_shape, dtype=np.float64)

    for oy in range(target_h):
        ys, ye = y0[oy], y1[oy]
        for ox in range(target_w):
            xs, xe = x0[ox], x1[ox]
            out[oy, ox] = img[ys:ye, xs:xe].mean(axis=(0, 1))

    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.rint(out), info.min, info.max)

    return out.astype(img.dtype, copy=False)
