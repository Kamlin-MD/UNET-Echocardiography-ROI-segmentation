.PHONY: install dev test lint clean train predict evaluate benchmark onnx

# Checkpoints and datasets are local resources, not supplied by a clone.
# See docs/RETRAINING.md; set MODEL_PATH for inference and evaluation.
MODEL_PATH ?= path/to/model.keras
IMAGE_DIR ?= data/images
MASK_DIR ?= data/masks

# ------- Installation -------
install:
	pip install -e .

dev:
	pip install -e ".[dev,notebooks]"

# ------- Quality -------
lint:
	ruff check echoroi/ tests/

format:
	ruff format echoroi/ tests/

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=echoroi --cov-report=term-missing --cov-report=html

# ------- Training -------
train:
	echoroi train \
		--image-dir data/images \
		--mask-dir data/masks \
		--model-path models/echoroi_unified.keras \
		--epochs 50 \
		--batch-size 8 \
		--learning-rate 1e-4 \
		--results-dir training_results

# ------- Inference -------
predict:
	@echo "Requires a local checkpoint: make predict MODEL_PATH=path/to/model.keras INPUT=path/to/image.png"
	echoroi predict \
		--model-path "$(MODEL_PATH)" \
		--input $(INPUT) \
		--output results/ \
		--visualize --deidentify

# ------- Evaluation -------
evaluate:
	echoroi evaluate \
		--model-path "$(MODEL_PATH)" \
		--image-dir "$(IMAGE_DIR)" \
		--mask-dir "$(MASK_DIR)" \
		--output evaluation_results

# ------- Benchmark -------
benchmark:
	@echo "Requires a local checkpoint: make benchmark MODEL_PATH=path/to/model.keras IMAGE=path/to/image.png"
	echoroi benchmark \
		--model-path "$(MODEL_PATH)" \
		--image-path $(IMAGE) \
		--num-runs 20

# ------- ONNX Conversion -------
onnx:
	@echo "Requires a local models/echoroi_unified.keras checkpoint; see docs/RETRAINING.md"
	python scripts/convert_to_onnx.py

# ------- Multi-env testing (tox) -------
tox:
	tox

tox-install-test:
	tox -e install-test

# ------- Clean -------
clean:
	rm -rf build/ dist/ *.egg-info/ .ruff_cache/ .mypy_cache/ .pytest_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
