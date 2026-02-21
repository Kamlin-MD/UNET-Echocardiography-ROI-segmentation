#!/usr/bin/env python3
"""Convert echoroi_unified.keras to ONNX and validate output parity."""

import os
import numpy as np

# 1. Load Keras model
print("Loading Keras model...")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import echoroi.model  # registers dice_coefficient & iou_score
model = tf.keras.models.load_model("models/echoroi_unified.keras")
print(f"  Input shape : {model.input_shape}")
print(f"  Output shape: {model.output_shape}")
print(f"  Parameters  : {model.count_params():,}")

# 2. Convert to ONNX
print("\nConverting to ONNX...")
import tf2onnx
import onnx

spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

onnx_path = "models/echoroi_unified.onnx"
onnx.save(onnx_model, onnx_path)
size_mb = os.path.getsize(onnx_path) / 1e6
print(f"  Saved to: {onnx_path} ({size_mb:.1f} MB)")

# 3. Validate with ONNX Runtime
print("\nValidating ONNX model...")
import onnxruntime as ort

sess = ort.InferenceSession(onnx_path)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print(f"  ONNX input : {input_name} {sess.get_inputs()[0].shape}")
print(f"  ONNX output: {output_name} {sess.get_outputs()[0].shape}")

# 4. Compare outputs on random input
print("\nComparing Keras vs ONNX outputs on random data...")
np.random.seed(42)
test_input = np.random.rand(1, 256, 256, 1).astype(np.float32)

keras_pred = model.predict(test_input, verbose=0)
onnx_pred = sess.run([output_name], {input_name: test_input})[0]

max_diff = np.max(np.abs(keras_pred - onnx_pred))
mean_diff = np.mean(np.abs(keras_pred - onnx_pred))
print(f"  Max  absolute diff: {max_diff:.2e}")
print(f"  Mean absolute diff: {mean_diff:.2e}")
assert max_diff < 1e-4, f"Output mismatch too large: {max_diff}"
print("  PASS — outputs match within tolerance")

# 5. Compare on real data sample
print("\nComparing on real image sample...")
from echoroi.preprocessing import UltrasoundPreprocessor
import glob

images = sorted(glob.glob("data/images/*.png"))[:4]
preprocessor = UltrasoundPreprocessor((256, 256))

for img_path in images:
    img = preprocessor.preprocess_image(img_path)
    img_batch = img[np.newaxis, ...]

    kp = model.predict(img_batch, verbose=0)
    op = sess.run([output_name], {input_name: img_batch.astype(np.float32)})[0]
    diff = np.max(np.abs(kp - op))
    print(f"  {os.path.basename(img_path)}: max_diff={diff:.2e}")
    assert diff < 1e-4

print("\nAll validations passed!")
print(f"Final artefacts in models/:")
for f in sorted(os.listdir("models")):
    sz = os.path.getsize(f"models/{f}") / 1e6
    print(f"  {f:40s} {sz:8.1f} MB")
