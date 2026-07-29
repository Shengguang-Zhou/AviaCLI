from __future__ import annotations

from pathlib import Path

from PIL import Image

_IMAGE_FORMAT_BY_SUFFIX = {
    ".bmp": "BMP",
    ".jpeg": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}


class ImageEncodingMismatch(ValueError):
    def __init__(self, *, expected_format: str, actual_format: str) -> None:
        self.expected_format = expected_format
        self.actual_format = actual_format
        super().__init__(
            "image encoding does not match its declared suffix: "
            f"expected={expected_format} actual={actual_format}"
        )


def decoded_image_size(path: Path) -> tuple[int, int]:
    expected_format = _IMAGE_FORMAT_BY_SUFFIX.get(path.suffix.lower())
    if expected_format is None:
        raise ValueError(f"image suffix is unsupported: {path.suffix!r}")
    try:
        with Image.open(path) as image:
            actual_format = str(image.format or "")
            if actual_format != expected_format:
                raise ImageEncodingMismatch(
                    expected_format=expected_format,
                    actual_format=actual_format or "unknown",
                )
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = int(image.width), int(image.height)
    except ImageEncodingMismatch:
        raise
    except Exception as exc:
        raise ValueError(f"cannot fully decode image: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return width, height


__all__ = ["ImageEncodingMismatch", "decoded_image_size"]
