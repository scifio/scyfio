# Tips for Claude and LLM Agents

This repo (`scyfio`) wraps the **SCIFIO** Java library (`io.scif`) via
`scyjava`/JPype, exposing a numpy/zarr/dask/xarray-friendly `BioFile` API. It is a
fork of the `bffile` project (which wrapped Bio-Formats directly); the Python data
layers (`_lazy_array`, `_zarr`, dask/xarray) are unchanged and ride on the
`core_metadata()` / `read_plane()` contract.

SCIFIO reaches Bio-Formats' full format coverage through **`scifio-bf-compat`**, which
adapts every Bio-Formats reader into a SCIFIO `Format`. So ND2/CZI/SVS/etc. still open —
through SCIFIO's axis-based API — while native formats (TIFF, GIF, …) use SCIFIO's own
readers.

## Testing tips

You can use `-n` to run in parallel:

```
uv run pytest -n 6
```

Test data is fetched on first run via `scripts/fetch_test_data.py` into `tests/data/`.

## Maven endpoints / JVM

Configured in `src/scyfio/_java_stuff.py`:
- `io.scif:scifio-bf-compat:4.1.1` (pulls scifio + scifio-ome-xml + formats-api)
- `ome:formats-gpl:6.10.1` (the actual Bio-Formats readers)
- `ch.qos.logback:logback-classic:1.3.15`

These resolve via the **scijava.public** Maven repo, not Maven Central. Override with
`SCIFIO_VERSION` / `FORMATS_VERSION` env vars. `get_scifio()` returns a cached
`io.scif.SCIFIO` context (the entry point for the reader initializer, translator
service, and format service).

## Understanding the underlying SCIFIO codebase

Local clones live under `~/code/scifio/`:
- `scifio/` — core API (`io.scif`)
- `scifio-bf-compat/` — Bio-Formats compatibility layer (`io.scif.bf`)
- `scifio-ome-xml/` — OME-XML translators/services (`io.scif.ome`)
- `scifio-tutorials/` — worked examples of the SCIFIO API

### Key SCIFIO API (and how scyfio uses it)

- **`io.scif.SCIFIO`** — context gateway. `scifio.initializer().initializeReader(loc,
  config)` returns an `io.scif.Reader`. The path must be wrapped in
  `org.scijava.io.location.FileLocation`, and a `io.scif.config.SCIFIOConfig` **must**
  be passed (the no-config path uses name-only format detection, which fails for
  content-detected formats like CZI → NullPointerException).
- **`io.scif.Reader`** — `openPlane(imageIndex, planeIndex, Interval)` → `io.scif.Plane`
  (`plane.getBytes()`). The `Interval` (`net.imglib2.FinalInterval.createMinSize(...)`)
  must span ALL planar axes (X, Y, and any interleaved colour-sample axis). `planeIndex`
  is a raster over the non-planar axes, computed via
  `io.scif.util.FormatTools.positionToRaster`. Lifecycle: `close(boolean fileOnly)`,
  `setSource`, `getCurrentLocation`. NB: a SCIFIO reader cannot reopen after
  `close(fileOnly)` — `scyfio` re-initializes a fresh reader on resume (see
  `BioFile._acquire_source`).
- **`io.scif.Metadata` / `io.scif.ImageMetadata`** — `meta.get(imageIndex)` →
  `ImageMetadata`; axes via `net.imagej.axis.Axes.{X,Y,Z,CHANNEL,TIME}`,
  `getPlanarAxisCount`, `getAxes`, `getAxisLength`, `getPixelType`, `isLittleEndian`,
  etc. Pixel types `io.scif.util.FormatTools.{INT8..DOUBLE}` (no `BIT`).
- **Pyramids** — SCIFIO has no resolution concept; bf-compat flattens Bio-Formats
  resolution levels into separate images. `scyfio` reconstructs `[series][resolution]`
  by peeking the wrapped Bio-Formats reader's `getCoreMetadataList()` (the per-series
  first entry keeps `resolutionCount`). All BF-peeking is encapsulated in
  `_biofile._bf_underlying_reader` / `_bf_resolution_counts`; non-BF formats get
  `resolution_count=1`.
- **OME-XML** — produced by translation:
  `scifio.translator().translate(meta, OMEMetadata, true)` then
  `omexml.getRoot().dumpXML()` (see `BioFile.ome_xml`).

### Where to look in the SCIFIO source

- Reader/Metadata contracts: `scifio/src/main/java/io/scif/{Reader,Metadata,ImageMetadata}.java`
- Pixel/index utilities: `scifio/src/main/java/io/scif/util/FormatTools.java`
- bf-compat reader + resolution peeking: `scifio-bf-compat/src/main/java/io/scif/bf/BioFormatsFormat.java`
- OME-XML translation: `scifio-ome-xml/src/main/java/io/scif/ome/`
