# Local checkpoints

This release includes neither pretrained weights nor source datasets.
Researchers must obtain source datasets independently from their providers.
Requests for sector annotations may be directed to the EchoROI authors;
availability depends on the applicable agreements and institutional
permissions.

The public software supports Keras checkpoints and ONNX exports. Cloning the
repository or installing the package does not supply either checkpoint.
Inference requires a locally trained or otherwise lawfully obtained model.
See the [training instructions](../docs/RETRAINING.md) and
[annotation-access policy](../docs/ANNOTATION_ACCESS.md).

`echoroi_unified.keras` and `echoroi_unified.onnx`, if present, are local,
ignored files, not release assets.
