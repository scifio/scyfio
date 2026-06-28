from __future__ import annotations

import contextlib
import logging
import os
import warnings
from functools import cache
from typing import Any

import jpype
import scyjava
import scyjava.config
from jpype.types import JString

# SCIFIO is reached through scifio-bf-compat, which adapts every Bio-Formats reader
# into a SCIFIO Format. We therefore load three endpoints:
#   - io.scif:scifio-bf-compat  (pulls scifio + scifio-ome-xml + formats-api)
#   - ome:formats-gpl           (the actual Bio-Formats readers; bf-compat only pulls
#                                formats-api, not the reader implementations)
#   - ch.qos.logback:logback-classic (logging backend; 1.3.x is the last Java-8 line)
# These resolve via the scijava.public Maven repo (not Maven Central).
SCIFIO_BF_COMPAT_COORDINATE = "io.scif:scifio-bf-compat:4.1.1"
FORMATS_GPL_COORDINATE = "ome:formats-gpl:6.10.1"

# Back-compat alias: some callers / tests still refer to MAVEN_COORDINATE.
MAVEN_COORDINATE = SCIFIO_BF_COMPAT_COORDINATE

# Configure Java constraints from environment variables
# BFF_JAVA_VENDOR: Java vendor (e.g., "zulu-jre", "adoptium", "temurin")
# BFF_JAVA_VERSION: Java version (e.g., "11", "17", "21")
# BFF_JAVA_FETCH: Fetch mode ("always", "never", "auto", default is "always")
_bff_vendor = os.getenv("BFF_JAVA_VENDOR") or None
_bff_version = os.getenv("BFF_JAVA_VERSION") or None
_bff_fetch = os.getenv("BFF_JAVA_FETCH") or None
if _bff_vendor or _bff_version:
    _kwargs = {}
    if _bff_vendor:
        _kwargs["vendor"] = _bff_vendor
    if _bff_version:
        _kwargs["version"] = _bff_version
    # Control fetch behavior via environment variable
    # Default: don't force download unless explicitly requested
    if _bff_fetch:
        _kwargs["fetch"] = _bff_fetch
    scyjava.config.set_java_constraints(**_kwargs)


def _resolve_coordinate(value: str, default: str) -> str:
    """Normalize a user-supplied Maven coordinate or bare version number."""
    # allow a single version number to be passed (applied to the default artifact)
    if ":" not in value and all(x.isdigit() for x in value.split(".") if x):
        group, artifact, _ = default.split(":", 2)
        return f"{group}:{artifact}:{value}"
    if not 2 <= len(value.split(":")) <= 5:
        warnings.warn(
            f"Invalid Maven coordinate env var: {value!r}. "
            "Must be a valid maven coordinate with 2-5 elements. "
            f"Using default {default!r}",
            stacklevel=2,
        )
        return default
    return value


# SCIFIO_VERSION overrides the scifio-bf-compat coordinate; BIOFORMATS_VERSION
# overrides the Bio-Formats readers (formats-gpl) coordinate. Each accepts either a
# bare version (e.g. "6.10.1") or a full Maven coordinate.
if _coord := os.getenv("SCIFIO_VERSION", ""):
    SCIFIO_BF_COMPAT_COORDINATE = _resolve_coordinate(
        _coord, SCIFIO_BF_COMPAT_COORDINATE
    )
    MAVEN_COORDINATE = SCIFIO_BF_COMPAT_COORDINATE
if _coord := os.getenv("BIOFORMATS_VERSION", ""):
    FORMATS_GPL_COORDINATE = _resolve_coordinate(_coord, FORMATS_GPL_COORDINATE)

scyjava.config.endpoints.append(SCIFIO_BF_COMPAT_COORDINATE)
scyjava.config.endpoints.append(FORMATS_GPL_COORDINATE)
# NB: logback 1.3.x is the last version with Java 8 support!
scyjava.config.endpoints.append("ch.qos.logback:logback-classic:1.3.15")

# #################################### LOGGING ####################################

# python-side logger

LOGGER = logging.getLogger("scyfio")
fmt = (
    "%(asctime)s.%(msecs)03d "  # timestamp with milliseconds
    "[%(levelname)-5s] "  # level, padded
    "%(name)s:%(lineno)d - "  # logger name and line no.
    "%(message)s"  # the log message
)
datefmt = "%Y-%m-%d %H:%M:%S"
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
# avoid double-logs if somebody has already attached handlers
if not any(isinstance(h, logging.StreamHandler) for h in LOGGER.handlers):
    LOGGER.addHandler(handler)


def redirect_java_logging(logger: logging.Logger | None = None) -> None:
    """Redirect Java logging to Python logger."""
    _logger = logger or LOGGER

    class PyAppender:
        def doAppend(self, event: Any) -> None:
            # event is an ILoggingEvent
            msg = str(event.getFormattedMessage())
            level = str(event.getLevel())
            # dispatch to Python logger
            getattr(logger, level.lower(), _logger.info)(msg)

        def getName(self) -> str:
            return "PyAppender"

    # Create a proxy for the Appender interface
    proxy = jpype.JProxy("ch.qos.logback.core.Appender", inst=PyAppender())

    # Get the LoggerContext
    Slf4jFactory = scyjava.jimport("org.slf4j.LoggerFactory")
    root = Slf4jFactory.getILoggerFactory().getLogger("ROOT")

    # remove the console appender
    with contextlib.suppress(AttributeError):
        for appender in root.iteratorForAppenders():
            if appender.getName() in ("console", "PyAppender"):
                root.detachAppender(appender)

        # add the Python appender
        root.addAppender(proxy)


@cache  # run only once
def start_jvm() -> None:
    """Start the JVM if not already running."""
    scyjava.start_jvm()  # won't repeat if already running
    redirect_java_logging()


@cache  # one shared SCIFIO context (a "lens" on a SciJava Context) per process
def get_scifio() -> Any:
    """Return a shared ``io.scif.SCIFIO`` context gateway.

    The SCIFIO instance is the entry point for discovering Formats and creating
    components (the reader initializer, the OME-XML translator service, the format
    service). It is created once and reused for the life of the process.
    """
    start_jvm()
    return scyjava.jimport("io.scif.SCIFIO")()


def jtype_to_python(obj: Any) -> Any:
    """Convert a Java type to a native Python type if possible."""
    if isinstance(obj, JString):
        return str(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, bool):
        return bool(obj)
    if hasattr(obj, "to_pint"):
        return obj.to_pint()
    return obj
