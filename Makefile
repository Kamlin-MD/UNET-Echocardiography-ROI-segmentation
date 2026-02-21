.PHONY: install dev test lint clean train predict evaluate benchmark onnx

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
	@echo "Usage: make predict INPUT=path/to/image.png"
	echoroi predict \
		--model-path models/echoroi_unified.keras \
		--input $(INPUT) \
		--output results/ \
		--visualize --deidentify

# ------- Evaluation -------
evaluate:
	echoroi evaluate \
		--model-path models/echoroi_unified.keras \
		--image-dir data/images \
		--mask-dir data/masks \
		--output evaluation_results

# ------- Benchmark -------
benchmark:
	@echo "Usage: make benchmark IMAGE=path/to/image.png"
	echoroi benchmark \
		--model-path models/echoroi_unified.keras \
		--image-path $(IMAGE) \
		--num-runs 20

# ------- ONNX Conversion -------
onnx:
	python scripts/convert_to_onnx.py

# ------- Clean -------
clean:
	rm -rf build/ dist/ *.egg-info/ .ruff_cache/ .mypy_cache/ .pytest_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
