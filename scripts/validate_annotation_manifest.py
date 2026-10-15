#!/usr/bin/env python3
"""Validate an authorised annotation package against locally reconstructed frames."""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath

from PIL import Image

FIELDS = (
    "dataset", "dataset_version", "source_relative_path", "frame_index",
    "frame_index_base", "source_height", "source_width", "extraction_recipe",
    "annotation_path", "annotation_sha256",
)
RECIPE_FIELDS = (
    "original_source_relative_path", "original_source_sha256",
    "decoded_frame_sha256", "decoder", "decoder_version", "frame_order",
    "orientation", "resizing", "color_conversion",
)


class ManifestError(ValueError):
    """A manifest or a referenced local file violates the matching contract."""


def local_file(root: Path, value: str, field: str) -> Path:
    """Resolve only existing files contained by an explicitly selected root."""
    relative = PurePosixPath(value)
    if (
        relative.is_absolute() or "\\" in value or ":" in value
        or ".." in relative.parts or not relative.parts
    ):
        raise ManifestError(f"{field}: expected a contained relative POSIX path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ManifestError(f"{field}: path escapes the selected root")
    if not path.is_file():
        raise ManifestError(f"{field}: file does not exist")
    return path


def integer(value: str, field: str, minimum: int = 0) -> int:
    if not re.fullmatch(r"[0-9]+", value) or int(value) < minimum:
        raise ManifestError(f"{field}: expected an integer >= {minimum}")
    return int(value)


def checksum(path: Path, expected: str, field: str, cache: dict[Path, str]) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ManifestError(f"{field}: expected a 64-digit SHA-256")
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        cache[path] = digest.hexdigest()
    if cache[path] != expected.lower():
        raise ManifestError(f"{field}: checksum mismatch")


def read_object(path: Path, field: str) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ManifestError(f"{field}: expected a JSON object")
    return value


def validate_png(path: Path, size: tuple[int, int], *, mask: bool = False) -> None:
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != size:
            raise ManifestError("PNG format or native dimensions do not match the manifest")
        if image.getexif().get(274, 1) != 1:
            raise ManifestError("PNG must retain native orientation without EXIF rotation")
        image.load()
        if mask:
            if image.mode not in ("1", "L"):
                raise ManifestError("mask must be single-channel binary PNG")
            colours = image.convert("L").getcolors(maxcolors=3)
            if colours is None or any(value not in (0, 255) for _, value in colours):
                raise ManifestError("mask pixels must be 0 or 255")


def validate_labelme(path: Path, source: Path, size: tuple[int, int]) -> None:
    annotation = read_object(path, "annotation_path")
    if annotation.get("imageData") is not None:
        raise ManifestError("LabelMe imageData must be absent or null; embedded images are forbidden")
    width, height = size
    if (
        type(annotation.get("imageWidth")) is not int
        or type(annotation.get("imageHeight")) is not int
        or (annotation["imageWidth"], annotation["imageHeight"]) != size
    ):
        raise ManifestError("LabelMe canvas dimensions do not match the manifest")
    if annotation.get("imagePath") != source.name:
        raise ManifestError("LabelMe imagePath must match the reconstructed frame basename")
    shapes = annotation.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ManifestError("LabelMe requires at least one ROI sector polygon")
    for shape in shapes:
        if (
            not isinstance(shape, dict) or shape.get("label") != "ROI sector"
            or shape.get("shape_type") != "polygon"
        ):
            raise ManifestError("LabelMe supports only ROI sector polygons")
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise ManifestError("LabelMe polygon must have at least three points")
        for point in points:
            if (
                not isinstance(point, list) or len(point) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
            ):
                raise ManifestError("LabelMe polygon coordinates must be finite numeric pairs")
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ManifestError("LabelMe polygon coordinates lie outside the native frame")


def validate_manifest(
    manifest: Path, source_root: Path, annotation_root: Path, original_root: Path,
) -> int:
    """Check matching integrity without printing restricted paths or decoding media."""
    source_root, annotation_root, original_root = (
        root.resolve() for root in (source_root, annotation_root, original_root)
    )
    for name, root in (
        ("source-root", source_root), ("annotation-root", annotation_root),
        ("original-root", original_root),
    ):
        if not root.is_dir():
            raise ManifestError(f"{name}: directory does not exist")
    seen_sources: set[Path] = set()
    seen_annotations: set[Path] = set()
    seen_frames: set[tuple[str, str, Path, int]] = set()
    hashes: dict[Path, str] = {}
    count = 0
    with manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames is None or (
            len(reader.fieldnames) != len(FIELDS) or set(reader.fieldnames) != set(FIELDS)
        ):
            raise ManifestError("header must contain exactly the ten documented fields")
        for row_number, row in enumerate(reader, start=2):
            try:
                if None in row or any(
                    not isinstance(row.get(field), str) or not row[field].strip()
                    for field in FIELDS
                ):
                    raise ManifestError("all ten fields must be present and nonempty")
                row = {field: row[field].strip() for field in FIELDS}
                index = integer(row["frame_index"], "frame_index")
                base = integer(row["frame_index_base"], "frame_index_base")
                if base not in (0, 1) or index < base:
                    raise ManifestError("frame index must be valid for an explicit base of 0 or 1")
                size = (
                    integer(row["source_width"], "source_width", 1),
                    integer(row["source_height"], "source_height", 1),
                )
                source = local_file(source_root, row["source_relative_path"], "source_relative_path")
                annotation = local_file(annotation_root, row["annotation_path"], "annotation_path")
                recipe_path = local_file(annotation_root, row["extraction_recipe"], "extraction_recipe")
                recipe = read_object(recipe_path, "extraction_recipe")
                if set(recipe) != set(RECIPE_FIELDS) or any(
                    not isinstance(recipe.get(field), str) or not recipe[field].strip()
                    for field in RECIPE_FIELDS
                ):
                    raise ManifestError("extraction_recipe must contain the ten documented string fields")
                for field, expected in (
                    ("frame_order", "presentation"), ("orientation", "native"), ("resizing", "none"),
                ):
                    if recipe[field] != expected:
                        raise ManifestError(f"extraction_recipe requires {field}={expected}")
                original = local_file(
                    original_root, recipe["original_source_relative_path"],
                    "original_source_relative_path",
                )
                pairing = (row["dataset"], row["dataset_version"], original, index - base)
                if source in seen_sources or annotation in seen_annotations or pairing in seen_frames:
                    raise ManifestError("duplicate reconstructed frame, annotation, or original-frame pairing")
                checksum(original, recipe["original_source_sha256"], "original_source_sha256", hashes)
                checksum(source, recipe["decoded_frame_sha256"], "decoded_frame_sha256", hashes)
                checksum(annotation, row["annotation_sha256"], "annotation_sha256", hashes)
                validate_png(source, size)
                if annotation.suffix.lower() == ".png":
                    validate_png(annotation, size, mask=True)
                elif annotation.suffix.lower() == ".json":
                    validate_labelme(annotation, source, size)
                else:
                    raise ManifestError("annotation_path must reference a PNG mask or LabelMe JSON")
                seen_sources.add(source)
                seen_annotations.add(annotation)
                seen_frames.add(pairing)
                count += 1
            except ManifestError as error:
                raise ManifestError(f"row {row_number}: {error}") from error
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ManifestError(
                    f"row {row_number}: could not read a referenced file as the required format"
                ) from error
    if not count:
        raise ManifestError("manifest has no annotation rows")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = validate_manifest(
            args.manifest, args.source_root, args.annotation_root, args.original_root,
        )
    except ManifestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError, csv.Error):
        print("ERROR: cannot read manifest as the required UTF-8 CSV", file=sys.stderr)
        return 1
    print(f"Validated {count} annotation pairings (integrity only; permissions not assessed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
