# Avia SDK

Reusable Python client primitives for Avia public APIs and dataset upload.

This package is intentionally small: it contains HTTP/auth helpers, local
dataset manifest scanning, signed upload transfer code, and import-session
operations. It must not import Avia backend application code, Runtime workers,
database clients, GPU libraries, vector stores, or curation algorithms.
