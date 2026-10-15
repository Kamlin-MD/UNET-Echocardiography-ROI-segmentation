"""Synthetic-only integrity checks for annotation packages."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from scripts.validate_annotation_manifest import FIELDS, ManifestError, validate_manifest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def package(tmp_path):
    source = tmp_path / "decoded"
    annotations = tmp_path / "annotations"
    originals = tmp_path / "originals"
    for root in (source, annotations, originals):
        root.mkdir()
    Image.new("L", (8, 6), 40).save(originals / "still.png")
    Image.new("L", (8, 6), 40).save(source / "frame.png")
    Image.new("L", (8, 6), 255).save(annotations / "mask.png")
    recipe = {
        "original_source_relative_path": "still.png",
        "original_source_sha256": sha(originals / "still.png"),
        "decoded_frame_sha256": sha(source / "frame.png"),
        "decoder": "Pillow; open native synthetic still and save PNG",
        "decoder_version": "synthetic-test",
        "frame_order": "presentation",
        "orientation": "native",
        "resizing": "none",
        "color_conversion": "none; native L uint8",
    }
    (annotations / "recipe.json").write_text(json.dumps(recipe))
    row = dict(zip(FIELDS, (
        "synthetic", "v1", "frame.png", "0", "0", "6", "8",
        "recipe.json", "mask.png", sha(annotations / "mask.png"),
    ), strict=True))
    return tmp_path / "manifest.csv", source, annotations, originals, row, recipe


def write_manifest(package, rows=None, fields=FIELDS):
    manifest, _, _, _, row, _ = package
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([row] if rows is None else rows)
    return manifest


def check(package):
    manifest, source, annotations, originals, _, _ = package
    return validate_manifest(manifest, source, annotations, originals)


def test_valid_png_and_cli(package):
    manifest = write_manifest(package)
    assert check(package) == 1
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/validate_annotation_manifest.py"),
         str(manifest), "--source-root", str(package[1]),
         "--annotation-root", str(package[2]), "--original-root", str(package[3])],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "Validated 1 annotation pairings" in result.stdout


@pytest.mark.parametrize(("field", "value", "message"), [
    ("dataset", "", "nonempty"),
    ("frame_index", "-1", "integer"),
    ("frame_index", "1.5", "integer"),
    ("frame_index_base", "2", "base"),
    ("frame_index_base", "1", "base"),
    ("source_width", "0", "integer"),
    ("source_height", "7", "dimensions"),
    ("source_relative_path", "missing.png", "does not exist"),
    ("annotation_path", "missing.png", "does not exist"),
    ("extraction_recipe", "missing.json", "does not exist"),
    ("source_relative_path", "../secret.png", "relative POSIX"),
    ("source_relative_path", "/secret.png", "relative POSIX"),
    ("annotation_path", "a\\mask.png", "relative POSIX"),
    ("annotation_sha256", "bad", "64-digit"),
    ("annotation_sha256", "0" * 64, "checksum mismatch"),
])
def test_invalid_fields(package, field, value, message):
    package[4][field] = value
    write_manifest(package)
    with pytest.raises(ManifestError, match=message):
        check(package)


def test_header_and_empty_manifest(package):
    write_manifest(package, rows=[], fields=FIELDS[:-1])
    with pytest.raises(ManifestError, match="header"):
        check(package)
    write_manifest(package, rows=[])
    with pytest.raises(ManifestError, match="no annotation rows"):
        check(package)


def test_duplicate_rows(package):
    write_manifest(package, rows=[package[4], package[4]])
    with pytest.raises(ManifestError, match="row 3: duplicate"):
        check(package)


def test_duplicate_original_frame_normalises_base(package):
    _, source, annotations, _, row, _ = package
    (source / "other.png").write_bytes((source / "frame.png").read_bytes())
    (annotations / "other.png").write_bytes((annotations / "mask.png").read_bytes())
    other = dict(row, source_relative_path="other.png", annotation_path="other.png",
                 frame_index="1", frame_index_base="1")
    write_manifest(package, rows=[row, other])
    with pytest.raises(ManifestError, match="duplicate"):
        check(package)


@pytest.mark.parametrize("field", ["original_source_sha256", "decoded_frame_sha256"])
def test_recipe_hash_mismatch(package, field):
    package[5][field] = "0" * 64
    (package[2] / "recipe.json").write_text(json.dumps(package[5]))
    write_manifest(package)
    with pytest.raises(ManifestError, match=f"{field}: checksum mismatch"):
        check(package)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("original_source_relative_path", "missing.avi", "does not exist"),
    ("orientation", "rotated", "orientation=native"),
    ("resizing", "256x256", "resizing=none"),
    ("frame_order", "decode", "frame_order=presentation"),
    ("decoder_version", "", "documented string fields"),
])
def test_invalid_recipe(package, field, value, message):
    package[5][field] = value
    (package[2] / "recipe.json").write_text(json.dumps(package[5]))
    write_manifest(package)
    with pytest.raises(ManifestError, match=message):
        check(package)


def test_corrupt_annotation_reports_row(package):
    path = package[2] / "mask.png"
    path.write_bytes(b"not a PNG")
    package[4]["annotation_sha256"] = sha(path)
    write_manifest(package)
    with pytest.raises(ManifestError, match="row 2: could not read"):
        check(package)


def test_source_dimensions_checked_independently(package):
    path = package[1] / "frame.png"
    Image.new("L", (9, 6), 40).save(path)
    package[5]["decoded_frame_sha256"] = sha(path)
    (package[2] / "recipe.json").write_text(json.dumps(package[5]))
    write_manifest(package)
    with pytest.raises(ManifestError, match="dimensions"):
        check(package)


def test_extra_or_missing_csv_values(package):
    manifest = write_manifest(package)
    with manifest.open("a") as stream:
        stream.write("synthetic,v1,frame.png\n")
    with pytest.raises(ManifestError, match="row 3: all ten fields"):
        check(package)
    with manifest.open("w") as stream:
        stream.write(",".join(FIELDS) + "\n")
        stream.write(",".join(package[4][key] for key in FIELDS) + ",extra\n")
    with pytest.raises(ManifestError, match="row 2: all ten fields"):
        check(package)


def test_symlink_escape(package, tmp_path):
    outside = tmp_path / "outside.png"
    Image.new("L", (8, 6)).save(outside)
    (package[1] / "escape.png").symlink_to(outside)
    package[4]["source_relative_path"] = "escape.png"
    write_manifest(package)
    with pytest.raises(ManifestError, match="escapes"):
        check(package)


@pytest.mark.parametrize(("size", "value", "message"), [
    ((8, 7), 255, "dimensions"),
    ((8, 6), 128, "0 or 255"),
])
def test_bad_mask(package, size, value, message):
    path = package[2] / "mask.png"
    Image.new("L", size, value).save(path)
    package[4]["annotation_sha256"] = sha(path)
    write_manifest(package)
    with pytest.raises(ManifestError, match=message):
        check(package)


def labelme(package):
    return {
        "imageWidth": 8, "imageHeight": 6, "imagePath": "frame.png",
        "imageData": None,
        "shapes": [{"label": "ROI sector", "shape_type": "polygon",
                    "points": [[0, 0], [7, 0], [4, 5]]}],
    }


def write_labelme(package, data):
    path = package[2] / "polygon.json"
    path.write_text(json.dumps(data))
    package[4].update(annotation_path="polygon.json", annotation_sha256=sha(path))
    write_manifest(package)


def test_valid_labelme(package):
    write_labelme(package, labelme(package))
    assert check(package) == 1


@pytest.mark.parametrize(("field", "value", "message"), [
    ("imageData", "embedded-source-image", "embedded images"),
    ("imageWidth", 9, "canvas dimensions"),
    ("imagePath", "other.png", "basename"),
    ("shapes", [], "at least one"),
    ("shapes", [{"label": "other"}], "ROI sector"),
    ("shapes", [{"label": "ROI sector", "shape_type": "polygon",
                 "points": [[0, 0], [8, 0], [2, 3]]}], "outside"),
    ("shapes", [{"label": "ROI sector", "shape_type": "polygon",
                 "points": [[0, 0], [float("nan"), 0], [2, 3]]}], "finite"),
])
def test_invalid_labelme(package, field, value, message):
    data = labelme(package)
    data[field] = value
    write_labelme(package, data)
    with pytest.raises(ManifestError, match=message):
        check(package)


def test_cli_failure_does_not_expose_source_paths(package):
    package[4]["source_relative_path"] = "restricted-patient-identifier.png"
    manifest = write_manifest(package)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/validate_annotation_manifest.py"),
         str(manifest), "--source-root", str(package[1]),
         "--annotation-root", str(package[2]), "--original-root", str(package[3])],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "row 2: source_relative_path: file does not exist" in result.stderr
    assert "restricted-patient-identifier" not in result.stderr
