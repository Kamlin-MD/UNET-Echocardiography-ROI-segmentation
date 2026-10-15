# Local checkpoints

Pretrained weights are not distributed with this release because the contributing datasets have differing access and reuse conditions. Researchers must obtain source datasets independently from their providers. Requests for scan-sector annotations should be directed to the EchoROI authors; availability is assessed for each dataset under the applicable agreements and institutional permissions.

The public software supports Keras checkpoints and ONNX exports. Cloning the
repository or installing the package does not supply either checkpoint.
Inference requires a locally trained or otherwise lawfully obtained model.
See the [training instructions](../docs/RETRAINING.md) and
[annotation-access policy](../docs/ANNOTATION_ACCESS.md).

`echoroi_unified.keras` and `echoroi_unified.onnx`, if present, are local,
ignored files, not release assets. Removing them from current Git tracking
does not remove historical GitHub/LFS downloads; see
[historical distribution](../CONTRIBUTING.md#historical-distribution-is-separate).
