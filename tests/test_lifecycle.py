"""Tests for ImageFile lifecycle: open / close / destroy state transitions."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
import pytest

from scyfio import ImageFile

if TYPE_CHECKING:
    from pathlib import Path


def _assert_uninitialized(bf: ImageFile) -> None:
    """Assert ImageFile is in UNINITIALIZED state."""
    assert bf.closed
    assert bf._java_reader is None
    assert bf._core_meta_list is None
    assert bf._suspended is False
    assert bf._finalizer is None


def _assert_open(bf: ImageFile) -> None:
    """Assert ImageFile is in OPEN state and can read data."""
    assert not bf.closed
    assert not bf.suspended
    assert bf._java_reader is not None
    assert bf._core_meta_list is not None
    plane = bf.read_plane()
    assert isinstance(plane, np.ndarray)


def _assert_suspended(bf: ImageFile) -> None:
    """Assert ImageFile is SUSPENDED: file handles closed, but reads still work.

    A read on a suspended file transparently re-acquires the source (scyfio
    re-initializes the SCIFIO reader), so reads work even when suspended. Metadata
    is preserved throughout.
    """
    assert bf.closed
    assert bf.suspended
    assert bf._java_reader is not None
    assert bf._core_meta_list is not None
    # Metadata works
    assert len(bf) > 0
    bf.core_metadata()
    plane = bf.read_plane()
    assert isinstance(plane, np.ndarray)

    # Data reads blocked
    # with pytest.raises(RuntimeError, match="not open"):
    #     bf.read_plane()


# ---------------------------------------------------------------------------
# UNINITIALIZED state
# ---------------------------------------------------------------------------


def test_uninitialized(simple_file: Path) -> None:
    """ImageFile starts UNINITIALIZED; all operations fail; close/destroy no-op."""
    bf = ImageFile(simple_file)
    _assert_uninitialized(bf)

    for method in (
        bf._ensure_java_reader,
        bf.read_plane,
        bf.as_array,
        bf.core_metadata,
    ):
        with pytest.raises(RuntimeError, match="not open"):
            method()  # type: ignore[call-arg]
    with pytest.raises(RuntimeError, match="not open"):
        len(bf)

    # close and destroy are safe no-ops
    bf.close()
    _assert_uninitialized(bf)
    bf.destroy()
    _assert_uninitialized(bf)


# ---------------------------------------------------------------------------
# UNINITIALIZED -> OPEN -> SUSPENDED -> OPEN -> ... -> destroy
# ---------------------------------------------------------------------------


def test_full_lifecycle(simple_file: Path) -> None:
    """Walk through every transition: open, close, reopen, destroy, reopen."""
    bf = ImageFile(simple_file)

    # UNINITIALIZED -> OPEN
    result = bf.open()
    assert result is bf  # returns self for chaining
    _assert_open(bf)
    reader_first = bf._java_reader
    meta_before = bf.core_metadata()

    # open() again is idempotent
    bf.open()
    assert bf._java_reader is reader_first

    # OPEN -> SUSPENDED
    bf.close()
    _assert_suspended(bf)
    assert bf.core_metadata() == meta_before

    # close() again is idempotent
    bf.close()
    _assert_suspended(bf)

    # SUSPENDED -> OPEN (re-initializes a fresh reader)
    bf.open()
    _assert_open(bf)

    # destroy() from OPEN -> UNINITIALIZED
    bf.destroy()
    _assert_uninitialized(bf)

    # destroy() is idempotent
    bf.destroy()
    _assert_uninitialized(bf)

    # Reopen from UNINITIALIZED (slow path — new reader)
    bf.open()
    _assert_open(bf)
    assert bf._java_reader is not reader_first

    # destroy() from SUSPENDED also works
    bf.close()
    _assert_suspended(bf)
    bf.destroy()
    _assert_uninitialized(bf)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager(simple_file: Path) -> None:
    """with block opens on enter, destroys on exit, supports re-entry."""
    bf = ImageFile(simple_file)

    # First context: open -> destroy
    with bf:
        _assert_open(bf)
        reader_first = bf._java_reader

        # close/open inside with block re-initializes the reader
        bf.close()
        _assert_suspended(bf)
        bf.open()
        _assert_open(bf)

    _assert_uninitialized(bf)  # __exit__ destroys

    # Re-enter: full re-init with new reader
    with bf:
        _assert_open(bf)
        assert bf._java_reader is not reader_first


def test_ensure_open(simple_file: Path) -> None:
    """ensure_open() suspends (not destroys), restores state, allows re-entry."""
    bf = ImageFile(simple_file)

    # Started closed -> ends suspended (vs direct context which destroys)
    with bf.ensure_open() as bf_inner:
        assert bf_inner is bf
        _assert_open(bf)
        meta = bf.core_metadata()

    _assert_suspended(bf)  # NOT destroyed like direct context manager
    assert bf.core_metadata() == meta

    # Restores previous state: started suspended -> ends suspended
    with bf.ensure_open():
        _assert_open(bf)
    _assert_suspended(bf)

    # Restores previous state: started open -> ends open
    bf.open()
    with bf.ensure_open():
        _assert_open(bf)
    _assert_open(bf)

    # Supports multiple re-entries with same reader (fast path)
    bf.close()
    for _ in range(3):
        with bf.ensure_open():
            _assert_open(bf)
        _assert_suspended(bf)


# ---------------------------------------------------------------------------
# GC finalizer
# ---------------------------------------------------------------------------


def test_gc_finalizer(simple_file: Path) -> None:
    """del bf triggers GC finalizer cleanup from both OPEN and SUSPENDED."""
    # From OPEN state
    bf = ImageFile(simple_file)
    bf.open()
    reader_open = bf._java_reader
    del bf
    gc.collect()
    assert reader_open is not None
    assert reader_open.getCurrentLocation() is None  # full cleanup

    # From SUSPENDED state
    bf = ImageFile(simple_file)
    bf.open()
    reader_suspended = bf._java_reader
    assert reader_suspended is not None
    bf.close()  # close(true) preserves the current location
    assert reader_suspended.getCurrentLocation() is not None
    del bf
    gc.collect()
    assert reader_suspended.getCurrentLocation() is None  # full cleanup


# ---------------------------------------------------------------------------
# Reader identity across the lifecycle
# ---------------------------------------------------------------------------


def test_reader_lifecycle(simple_file: Path) -> None:
    """Suspend/resume and destroy/reopen both yield a working reader.

    SCIFIO readers cannot reopen after ``close(fileOnly)``, so resuming
    re-initializes a fresh reader (rather than reusing the suspended one).
    """
    bf = ImageFile(simple_file)
    bf.open()

    # Suspend/resume: reads work again afterwards
    bf.close()
    bf.open()
    _assert_open(bf)

    # Destroy + reopen
    bf.destroy()
    bf.open()
    _assert_open(bf)

    bf.destroy()
