---
icon: lucide/rocket
title: Get started
---

# Getting started with `scyfio`

`scyfio` is a **modern Bio-Formats wrapper for Python**.

It exposes the full power of Bio-Formats in a clean Pythonic API
backed by lazy data access.

### Features

- [Full lazy indexing and slicing](./usage.md#reading-data-with-lazybioarray)
  (with no additional dependencis on dask/xarray/zarr) with
  [`scyfio.LazyBioArray`][]
- [Custom zarr store](./usage.md#complete-virtual-ome-zarr-view) presents files
  as complete multi-resolution OME-Zarr group
- [Export to xarray
  DataArray](./usage.md#labeled-dimensionscoordinates-with-xarray)  with
  metadata-aware dimension and coordinate labels
- [Export to Dask arrays](./usage.md#lazy-computation-with-dask) for parallel
  and out-of-core computation

!!! tip "Batteries included"
    **No special environment setup is required**, thanks to
    [`scyjava`](https://github.com/scijava/scyjava),
    [`jgo`](https://github.com/apposed/jgo),
    [`jpype`](https://github.com/jpype-project/jpype), and
    [`cjdk`](https://github.com/cachedjdk/cjdk):

    just `pip install scyfio` and you're ready to go.

## Installation

```bash
pip install scyfio
```

Optional extras for zarr, xarray, and dask support, include:

```bash
pip install scyfio[zarr,xarray,dask]
```

## Quick start

```python
import scyfio

# load directly into memory
data = scyfio.imread("path/to/file", series=0)

# or use lazy access to load only what you need
with scyfio.BioFile("path/to/file") as biofile:
    # lazy series accessor
    lazy_array = biofile[0].as_array()
    # Load data into memory (T=0, C=1:4, Z=all, Y=100:200)
    data = lazy_array[0, 1:4, :, 100:200]
```

See [usage](usage.md) for more details and examples.
