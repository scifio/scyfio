# scyfio

[![License](https://img.shields.io/pypi/l/scyfio.svg?color=green)](https://github.com/imaging-formats/scyfio/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/scyfio.svg?color=green)](https://pypi.org/project/scyfio)
[![Python Version](https://img.shields.io/pypi/pyversions/scyfio.svg?color=green)](https://python.org)
[![CI](https://github.com/imaging-formats/scyfio/actions/workflows/ci.yml/badge.svg)](https://github.com/imaging-formats/scyfio/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imaging-formats/scyfio/branch/main/graph/badge.svg)](https://codecov.io/gh/imaging-formats/scyfio)

Modern SCIFIO-based scientific image reader for Python

`scyfio` wraps the [SCIFIO](https://scif.io) Java library. SCIFIO reads many
formats natively and reaches Bio-Formats's full format coverage through
[`scifio-bf-compat`](https://github.com/scifio/scifio-bf-compat), which adapts
every Bio-Formats reader into a SCIFIO format.

Fork of [bffile](https://github.com/tlambert03/bffile).

### Documentation 📖

<https://imaging-formats.github.io/scyfio/>

## Installation

```bash
pip install scyfio
```

## Usage

### Quick Start

```python
from scyfio import ImageFile
import numpy as np

with ImageFile("tests/data/ND2_dims_p4z5t3c2y32x32.nd2") as bf:
    # Access OME metadata
    print(bf.ome_metadata)  # ome_types.OME object

    # Get lazy array for series 0 (no data loaded yet)
    arr = bf.as_array(series=0)
    print(arr.shape, arr.dtype)  # (T, C, Z, Y, X) shape

    # Index specific planes on-demand
    plane = arr[0, 0, 2]  # Only reads this plane
    roi = arr[:, :, :, 100:200, 50:150]  # Sub-regions

    # Materialize all data when needed
    full_data = np.array(arr)

    # Or use dask for lazy computation
    darr = bf.to_dask(series=0, chunks="auto")
    result = darr.mean(axis=2).compute()
```

For simple cases, use `imread()` to load a single series/resolution into memory:

```python
from scyfio import imread

# Read series0/resolution0 into numpy array
# series and resolution parameters are optional and default to 0
data = imread("image.nd2", series=0, resolution=0)  
print(data.shape, data.dtype)  # (T, C, Z, Y, X) array
```

### Selecting Java library versions

The Java libraries are downloaded at runtime (via [jgo](https://pypi.org/project/jgo/)).
By default scyfio loads two coordinates plus a logging backend:

- `io.scif:scifio-bf-compat` — SCIFIO and its Bio-Formats compatibility layer
- `ome:formats-gpl` — the actual Bio-Formats readers

You can override either with an environment variable. Each accepts a simple
version string (e.g. `6.10.1`) or a full maven coordinate:

```bash
# Override the SCIFIO / scifio-bf-compat version
export SCIFIO_VERSION="io.scif:scifio-bf-compat:4.1.1"

# Override the Bio-Formats readers version
export BIOFORMATS_VERSION="6.10.1"
```

> [!IMPORTANT]
> `scifio-bf-compat` is built against a specific Bio-Formats API version, so the
> `BIOFORMATS_VERSION` you choose must stay API-compatible with it. Mismatched
> versions can fail at runtime (e.g. `NoSuchFieldError`). When in doubt, leave the
> defaults alone.

To see the loaded SCIFIO version and the maven coordinate that was used:

```python
from scyfio import ImageFile

print(ImageFile.scifio_version())      # e.g. "0.39.1"
print(ImageFile.maven_coordinate())    # e.g. "io.scif:scifio-bf-compat:4.1.1"
```

### Java Runtime

> [!TIP]
> **No manual Java installation required!**  
>
> This package automatically downloads and manages the Java runtime using
> [cjdk](https://github.com/cachedjdk/cjdk) (via [scyjava](https://github.com/scijava/scyjava)).

By default, scyjava uses **Zulu JRE 11** (defined
[here](https://github.com/scijava/scyjava/blob/a2b8bed0a07a87d4c9b715a6dddaad308080f440/src/scyjava/config.py#L15-L20)).
You can configure the Java version and/or vendor using environment variables:

```bash
# Use Adoptium JDK 17
export SCYFIO_JAVA_VERSION=17
export SCYFIO_JAVA_VENDOR=adoptium

# Use Temurin JDK 21
export SCYFIO_JAVA_VERSION=21
export SCYFIO_JAVA_VENDOR=temurin
```

Available vendors: `zulu-jre`, `zulu`, `adoptium`, `temurin`, and
[other vendors listed in cjdk](https://github.com/cachedjdk/cjdk)

#### Java 8 Support

scyfio is *not* currently expected to work with Java 8, as `jpype` has
deprecated support for Java 8 as of `jpype` version 1.6. If you do want to try
with Java 8, you will minimally need to explicitly pin `jpype<1.6`.  If this is
an important use case for you, please open an issue to discuss Java 8 support.

## License

Licensing is a bit complicated for this project, so please read carefully.

- All code in this `scyfio` repository is licensed under the [BSD-3-Clause
  License](./LICENSE/LICENSE).  You may use and distribute this code under the
  terms of the BSD-3-Clause License.
- However, a user who installs `scyfio` from PyPI will, by default, end up with
  Java jars that are licensed under the GPL-2.0 License (read below). As such,
  **this package is listed on PyPI as GPL-2.0-or-later**.

When you run scyfio for the first time, it will automatically download a number
of Java jars (via [`jgo`](https://pypi.org/project/jgo/)), each of which has its
own license. SCIFIO and its compatibility layer
([`io.scif:scifio-bf-compat`](https://github.com/scifio/scifio-bf-compat)) are
BSD-licensed, but by default scyfio also downloads the
[`ome:formats-gpl`](https://mvnrepository.com/artifact/ome/formats-gpl) Bio-Formats
readers.

- `ome:formats-gpl` is licensed under the [GPLv2+
  License](./LICENSE/LICENSE_FORMATS_GPL)

If you would like to use scyfio without any GPL-licensed jars, you can instead
opt into the BSD-licensed
[`ome:formats-bsd`](https://mvnrepository.com/artifact/ome/formats-bsd) readers by
setting the `BIOFORMATS_VERSION` environment variable. Use a version that is
API-compatible with the bundled `scifio-bf-compat` (matching the default
`ome:formats-gpl` version):

```
BIOFORMATS_VERSION="ome:formats-bsd:6.10.1"
```

- `ome:formats-bsd` is licensed under the [BSD-2-Clause
  License](./LICENSE/LICENSE_FORMATS_BSD)

If you need a package that defaults to BSD-licensed jars (and ships as
BSD-3 on PyPI), open an issue to discuss a `scyfio-bsd` variant.
