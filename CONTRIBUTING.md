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
- **`paper/`** — Unrefereed technical manuscript source and bibliography.

## Zenodo Release Checklist

1. Select the release version and date. Keep `pyproject.toml`, `CITATION.cff`,
   the package version, and the software citation examples consistent with
   that release; model versions are separate from package versions. Do not
   reuse an existing tag for changed contents.
2. Run the existing tests and lint checks. Review the manuscript sources,
   bibliography, and Word export using [the manuscript instructions](paper/README.md).
3. Review the exact files to be archived for identifiers, institutional
   permissions, and third-party dataset redistribution terms. Local training
   data is not necessarily part of the tracked release. The software's MIT
   licence does not override source-dataset licences or access restrictions;
   do not add restricted patient data to a public archive.
4. Connect the repository to Zenodo's GitHub integration and enable archiving
   before publishing the GitHub release. Zenodo supports the existing
   `CITATION.cff` for software metadata. Check the imported title, all four
   authors, ORCID, description, licence, version, and release date. If a
   `.zenodo.json` is introduced later, Zenodo will use it instead of
   `CITATION.cff`, so the metadata must then be kept in sync.
5. Inspect the exact release tree, source archive, wheel/sdist, attached
   assets, and Zenodo deposit. They must exclude checkpoints (including LFS
   pointers), restricted source images, unapproved annotation packages,
   LabelMe embedded `imageData`, and disallowed notebook outputs. Clear
   notebook outputs before release. Review any retained manuscript figures
   and reference prediction grids for permission and identifiers separately;
   do not infer permission from their presence in Git.
6. Publish the release and verify the resulting Zenodo record and downloadable
   contents. Record the version-specific DOI in `CITATION.cff` and the
   README/model-card citation examples once assigned, using Zenodo's reserved
   DOI workflow if it must be included in the tagged sources. Otherwise,
   update these files for subsequent releases without rewriting the old tag.
   Distinguish the version DOI from the concept DOI, which covers all versions.

The technical manuscript may accompany the software archive, but remains
unrefereed. If deposited separately as a technical report, link the two records
and keep the software and report citations distinct.

Use the [annotation-access policy](docs/ANNOTATION_ACCESS.md) for new requests.
Source-data access does not automatically authorise annotation distribution.

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
