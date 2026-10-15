# Annotation-only packages and matching convention

No actual annotations are committed here. Only this policy and a synthetic
manifest template are public. Packages remain ignored until their per-source
distribution terms have been assessed; do not force-add them.
See [annotation access](../docs/ANNOTATION_ACCESS.md) before requesting files.

Each authorised package must state its allowed recipients, purposes, source
dataset version, redistribution rules, and any institutional restrictions.
Permission for one source does not apply to the others. Do not embed source
images, cine/DICOM data, or LabelMe `imageData` (omit it or set it to `null`).
Do not put restricted source identifiers in public examples, commits, logs,
or notebook outputs; retain them in the authorised package where required.

## Manifest contract

The UTF-8 CSV header must contain exactly:

```text
dataset,dataset_version,source_relative_path,frame_index,frame_index_base,source_height,source_width,extraction_recipe,annotation_path,annotation_sha256
```

All fields are required. Use forward-slash relative paths without `..`;
absolute paths, backslashes, and symlinks escaping the selected roots are
rejected. Path matching is case-sensitive on a case-sensitive filesystem.

- `dataset` and `dataset_version`: provider name and exact agreed release
  identifier, not a guessed paper year.
- `source_relative_path`: path under `--source-root` to a **locally
  reconstructed, lossless PNG frame**, not the original video/DICOM file.
  Keep dimensions and orientation native. Each decoded frame path must occur
  once. Use source-specific subdirectories to avoid collisions.
- `frame_index`: integer index in the original decoded sequence;
  `frame_index_base` is explicitly `0` or `1`. Normalised index is
  `frame_index - frame_index_base`. For an original still, use `0,0`.
  Indices refer to decoded presentation order, not filenames, timestamps,
  sampled/stride order, or a representative-frame heuristic.
- `source_height`, `source_width`: positive integer dimensions of the
  native decoded frame. Both the reconstructed PNG and annotation canvas
  must match these values, before training resize/padding.
- `extraction_recipe`: path under `--annotation-root` to a per-frame JSON
  recipe described below. A recipe can disclose restricted source identifiers
  and must stay in the authorised package.
- `annotation_path`: path under `--annotation-root` to either a single-channel
  binary PNG mask (values 0 and 255, or a 1-bit PNG), or a LabelMe JSON polygon
  annotation. Each annotation path must occur once.
- `annotation_sha256`: SHA-256 of the exact annotation file bytes, 64 hex
  digits. Checksums establish identity/integrity, not permission or correctness.

## Decoding recipe and coordinates

Each recipe is a JSON object with nonempty string fields:

| Field | Meaning |
|---|---|
| `original_source_relative_path` | Original still/video/DICOM path under `--original-root` |
| `original_source_sha256` | SHA-256 of that original file |
| `decoded_frame_sha256` | SHA-256 of the reconstructed PNG bytes |
| `decoder` | Exact library/tool and decoding command or procedure |
| `decoder_version` | Library/tool version, plus relevant codec/plugin versions |
| `frame_order` | Must be `presentation`; specify index as in the manifest |
| `orientation` | Must be `native`: no EXIF rotation, display transform, flip, or mirroring |
| `resizing` | Must be `none`: no crop, resize, padding, or interpolation before annotation |
| `color_conversion` | Exact colour/photometric and bit-depth conversion, including range/windowing |

Use the agreed decoder/codec and versions to decode every frame in presentation
order, then select the specified index. Do not substitute random-access seek,
temporal subsampling, or highest-entropy frame selection. For DICOM, specify
transfer-syntax plugin, `pixel_array` frame ordering, planar/colour conversion
(including YBR or palette handling), MONOCHROME1 inversion if any, and window/
range conversion in the recipe; no unspecified automatic display transforms.
Document how higher bit depths are mapped to PNG. Do not assume decoding by
another backend produces identical pixels or file bytes. Agreement on the
recipe and hashes is required before matching.

Coordinates are in the native, orientation-preserved frame: `(x, y)` means
column from the left and row from the top, with integer pixel centres starting
at `(0, 0)`. LabelMe `imageWidth`/`imageHeight` must match the manifest and
`imagePath` must be the reconstructed frame's basename. The supported shapes
are polygons with label `ROI sector`, at least three finite points, and
`0 <= x < width`, `0 <= y < height`. The existing converter truncates polygon
coordinates to integers (`np.int32`) and fills their union using OpenCV
`fillPoly`; preserve that convention. A raster mask must use these same
coordinates and dimensions; do not distribute a resized 256 x 256 mask for
a larger native frame without a separately agreed transform.

The validator checks existence, hashes, native dimensions, permitted annotation
format/coordinates, and unique original-frame pairings (normalising index base).
It **does not decode the original media or prove the recipe was executed**.
Reconstruct first using the agreed recipe, then verify the resulting PNG hash.
If legacy annotations used different transforms, obtain an explicit corrected
recipe/package from the authors rather than silently rescaling them.

## Local integrity check

```bash
python scripts/validate_annotation_manifest.py authorised/manifest.csv \
  --source-root local/decoded_frames \
  --original-root local/provider_downloads \
  --annotation-root authorised
```

All roots and referenced files must exist. Empty manifests, incomplete rows,
duplicate pairings, missing files, mismatched sizes/hashes, invalid polygons,
and embedded source images fail with a nonzero exit status and a row number.
The validator avoids echoing restricted source paths in its error messages.
The synthetic `manifest.example.csv` has intentionally fictitious paths and
hashes; it is not expected to validate against any real dataset.
