@_default:
    @just --list

# build java stubs into ./typings
build-stubs:
    uv run scripts/stubgen.py 'io.scif:scifio-bf-compat:4.1.1' 'ome:formats-gpl:6.10.1' --prefix io.scif --prefix loci --prefix ome --prefix java

# run tests quickly with coverage
test *ARGS:
    uv run pytest -n 6 --cov --cov-report=xml --cov-report=term-missing {{ARGS}}
    uv run diff-cover coverage.xml --compare-branch=upstream/main

# run linting and type checking
check:
    uv run prek -a --hook-stage=manual

# clone the SCIFIO repositories (for understanding the underlying Java code)
clone-scifio:
    git clone https://github.com/scifio/scifio
    git clone https://github.com/scifio/scifio-bf-compat
    git clone https://github.com/scifio/scifio-ome-xml
    git clone https://github.com/scifio/scifio-tutorials
