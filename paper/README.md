# EchoROI technical manuscript

`paper.md` and `paper.bib` are the manuscript source and bibliography.
The manuscript is unrefereed and retained as material for a Biomedical
Engineering doctoral thesis chapter at Stellenbosch University. The published
external MIMIC dataset manuscript is a separate thesis contribution.

The manuscript uses a neutral layout without a journal-specific build or
submission workflow. `paper.docx` is generated from the source and bibliography.

This release includes neither pretrained weights nor source datasets.
Researchers must obtain source datasets independently from their providers.
Requests for scan-sector annotations may be directed to the EchoROI authors;
availability depends on the applicable agreements and institutional
permissions.

See [annotation access](../docs/ANNOTATION_ACCESS.md),
[retraining](../docs/RETRAINING.md), and the
[run-record reconciliation](../training_results/README.md). The manuscript
preserves the original reported results while qualifying the collection/run
count discrepancy and the limits on exact checkpoint reproduction.

To regenerate it from the repository root with Pandoc:

```bash
cd paper
pandoc paper.md --from=markdown --standalone --citeproc \
  --resource-path=. --fail-if-warnings --output=paper.docx
```

Review the generated document before release or thesis incorporation. Any local `paper.pdf` is ignored and is not part of the tracked release. A
neutral PDF build remains to be verified.

A Zenodo software record archives a specific software release, which can
include this source manuscript. It does not imply peer review. A separate
technical-report deposit and DOI are optional and should be linked to the
software record if created.
