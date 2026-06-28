from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from ._jimports import jimport

if TYPE_CHECKING:
    import loci.formats
    from typing_extensions import Self


class OMEShape(NamedTuple):
    """NamedTuple with OME metadata shape."""

    t: int
    c: int
    z: int
    y: int
    x: int
    rgb: int

    @property
    def as_array_shape(self) -> tuple[int, ...]:
        """Return 5-tuple `(T,C,Z,Y,X)` if `rgb==1`, else 6-tuple `(T,C,Z,Y,X,RGB)`."""
        if self.rgb > 1:
            return (self.t, self.c, self.z, self.y, self.x, self.rgb)
        return (self.t, self.c, self.z, self.y, self.x)

    def __repr__(self) -> str:
        return (
            f"(t={self.t}, c={self.c}, z={self.z}, "
            f"y={self.y}, x={self.x}, rgb={self.rgb})"
        )


def pixtype2dtype(pixeltype: int, little_endian: bool) -> np.dtype:
    """Convert a SCIFIO pixel type integer into a numpy dtype."""
    FormatTools = jimport("io.scif.util.FormatTools")

    fmt2type: dict[int, str] = {
        FormatTools.INT8: "i1",
        FormatTools.UINT8: "u1",
        FormatTools.INT16: "i2",
        FormatTools.UINT16: "u2",
        FormatTools.INT32: "i4",
        FormatTools.UINT32: "u4",
        FormatTools.FLOAT: "f4",
        FormatTools.DOUBLE: "f8",
    }
    return np.dtype(("<" if little_endian else ">") + fmt2type[pixeltype])


@dataclass
class CoreMetadata:
    """Core metadata for a single series."""

    dtype: np.dtype
    shape: OMEShape
    rgb_count: int = 0
    thumb_size_x: int = 0
    thumb_size_y: int = 0
    bits_per_pixel: int = 0
    image_count: int = 0
    modulo_z: Any = None
    modulo_c: Any = None
    modulo_t: Any = None
    dimension_order: str = ""
    is_order_certain: bool = False
    is_rgb: bool = False
    is_little_endian: bool = False
    is_interleaved: bool = False
    is_indexed: bool = False
    is_false_color: bool = True
    is_metadata_complete: bool = False
    is_thumbnail_series: bool = False
    series_metadata: dict[str, Any] = field(default_factory=dict)
    resolution_count: int = 1

    @classmethod
    def from_java(cls, meta: loci.formats.CoreMetadata) -> Self:

        if size_zt := meta.sizeZ * meta.sizeT:
            eff_size_c = meta.imageCount // size_zt
        else:
            eff_size_c = 1

        if eff_size_c == 0:
            rgb_count = 1
        else:
            rgb_count = meta.sizeC // eff_size_c

        return cls(
            dtype=pixtype2dtype(meta.pixelType, meta.littleEndian),
            shape=OMEShape(
                x=meta.sizeX,
                y=meta.sizeY,
                z=meta.sizeZ,
                c=eff_size_c,
                t=meta.sizeT,
                rgb=rgb_count,
            ),
            rgb_count=rgb_count,
            thumb_size_x=meta.thumbSizeX,
            thumb_size_y=meta.thumbSizeY,
            bits_per_pixel=meta.bitsPerPixel,
            image_count=meta.imageCount,
            modulo_z=meta.moduloZ,
            modulo_c=meta.moduloC,
            modulo_t=meta.moduloT,
            dimension_order=str(meta.dimensionOrder),
            is_order_certain=meta.orderCertain,
            is_rgb=meta.rgb,
            is_little_endian=meta.littleEndian,
            is_interleaved=meta.interleaved,
            is_indexed=meta.indexed,
            is_false_color=meta.falseColor,
            is_metadata_complete=meta.metadataComplete,
            is_thumbnail_series=meta.thumbnail,
            series_metadata=dict(meta.seriesMetadata),
            resolution_count=meta.resolutionCount,
        )

    @classmethod
    def from_image_metadata(cls, imeta: Any) -> Self:
        """Build CoreMetadata from a native SCIFIO ``io.scif.ImageMetadata``.

        Used for formats served by SCIFIO's own readers (rather than the Bio-Formats
        compatibility layer). SCIFIO describes images with a flexible list of
        ``net.imagej.axis`` axes split into "planar" axes (which define the physical
        plane, e.g. X, Y, and any interleaved colour samples) and "non-planar" axes
        (which enumerate planes, e.g. Z, Channel, Time). We collapse that model down to
        the fixed ``(T, C, Z, Y, X[, rgb])`` shape the rest of scyfio expects.
        """
        Axes = jimport("net.imagej.axis.Axes")

        def axis_len(axis_type: Any, default: int) -> int:
            if imeta.getAxisIndex(axis_type) < 0:
                return default
            return int(imeta.getAxisLength(axis_type))

        x = axis_len(Axes.X, 1)
        y = axis_len(Axes.Y, 1)
        z = axis_len(Axes.Z, 1)
        t = axis_len(Axes.TIME, 1)

        # rgb = product of planar axis lengths other than X and Y (the colour samples
        # packed into each plane, e.g. interleaved RGB).
        planar_count = imeta.getPlanarAxisCount()
        rgb_count = 1
        for d in range(planar_count):
            axis_type = imeta.getAxis(d).type()
            if axis_type != Axes.X and axis_type != Axes.Y:
                rgb_count *= int(imeta.getAxisLength(axis_type))
        rgb_count = max(rgb_count, 1)

        # effective channel count: total planes divided by the Z*T planes, mirroring
        # Bio-Formats' notion of "effective sizeC" (handles extra channel-like axes).
        plane_count = int(imeta.getPlaneCount())
        size_zt = z * t
        eff_size_c = (plane_count // size_zt) if size_zt else 1
        eff_size_c = max(eff_size_c, 1)

        # dimension order string, e.g. "XYCZT", from the axis list
        order = "".join(
            str(imeta.getAxis(d).type().getLabel())[:1].upper()
            for d in range(imeta.getAxes().size())
        )

        return cls(
            dtype=pixtype2dtype(imeta.getPixelType(), imeta.isLittleEndian()),
            shape=OMEShape(x=x, y=y, z=z, c=eff_size_c, t=t, rgb=rgb_count),
            rgb_count=rgb_count,
            thumb_size_x=int(imeta.getThumbSizeX()),
            thumb_size_y=int(imeta.getThumbSizeY()),
            bits_per_pixel=int(imeta.getBitsPerPixel()),
            image_count=plane_count,
            dimension_order=order,
            is_order_certain=bool(imeta.isOrderCertain()),
            is_rgb=rgb_count > 1,
            is_little_endian=bool(imeta.isLittleEndian()),
            is_interleaved=imeta.getInterleavedAxisCount() > 0,
            is_indexed=bool(imeta.isIndexed()),
            is_false_color=bool(imeta.isFalseColor()),
            is_metadata_complete=bool(imeta.isMetadataComplete()),
            is_thumbnail_series=bool(imeta.isThumbnail()),
            resolution_count=1,
        )
