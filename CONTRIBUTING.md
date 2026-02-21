# Contributing to EchoROI

Thank you for your interest in contributing to EchoROI!

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install in development mode:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,notebooks]"
   ```

3. Create a feature branch:

   ```bash
   git checkout -b feature/my-improvement
   ```

## Development Workflow

### Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
make lint      # check for issues
make format    # auto-format
```

### Running Tests

```bash
make test          # quick run
make test-cov      # with coverage report
```

All pull requests must pass the existing test suite. Please add tests
for any new functionality.

### Project Structure

- **`echoroi/`** — The installable Python package (model, training,
  inference, preprocessing, CLI).
- **`tests/`** — pytest test suite.
- **`scripts/`** — Utility scripts (not part of the package).
- **`notebooks/`** — Jupyter notebooks for exploration.
- **`paper/`** — JOSS manuscript source.

## Submitting Changes

1. Ensure all tests pass and `make lint` is clean.
2. Write clear, concise commit messages.
3. Open a pull request against `main` with a description of what
   changed and why.

## Reporting Issues

Open an issue on GitHub with:

- A clear title and description.
- Steps to reproduce (if applicable).
- Your Python version and OS.

## Adding Training Data

If you have annotated echocardiogram frames you would like to
contribute, please open an issue to discuss. Due to patient privacy
requirements, we cannot accept data that may contain protected health
information.

## Code of Conduct

Be respectful and constructive. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/) code of
conduct.
