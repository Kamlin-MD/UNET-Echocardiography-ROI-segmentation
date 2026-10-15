# Source datasets and scan-sector annotation access

This release includes neither pretrained weights nor source datasets.
Researchers must obtain each source dataset independently from its provider.
Requests for scan-sector annotations should be directed to the EchoROI
authors; availability is assessed for each dataset under the applicable
agreements and institutional permissions.

Contact **kamlin.ekambaram@gmail.com** with your institution, intended use,
dataset names, exact provider release/version, access authorisation, and the
requested subset. Do not email patient images or restricted identifiers
through an unauthorised channel. The authors will arrange an appropriate
channel if a request can proceed.

Source-data access does **not** automatically guarantee annotation access.
No actual annotations are included in this release, and no request is
guaranteed approval. The software's MIT licence does not license the source
datasets or separately authorised annotation packages.

## Per-source request information

The following is the documented collection inventory, not verified membership
of the saved 1,206-pair training run. Unless identified below, the precise
provider version used by the authors is not established in the released run
records; it must be resolved during a request, not guessed from the paper year.

| Dataset (inventory frames) | Provider access route | Version to specify in a request | Scan-sector annotation status |
|---|---|---|---|
| MIMIC-IV-ECHO (403) | [PhysioNet record](https://physionet.org/content/mimic-iv-echo/1.0/); complete provider credentialing and agreements | The manuscript cites 1.0; confirm the exact files/subset with the authors | Not publicly distributed here; assess against PhysioNet agreements and institutional permissions |
| EchoNet-Dynamic (145) | [Stanford EchoNet-Dynamic](https://echonet.github.io/dynamic/); obtain access under the provider's current terms | Exact provider release/download date and file checksums; author-run version not recorded | Not publicly distributed here; per-request assessment under applicable agreements |
| EchoNet-Paediatric (263) | [Stanford EchoNet-Pediatric](https://echonet.github.io/pediatric/); follow provider access instructions | Exact paediatric release, view subset, download date, and file checksums; author-run version not recorded | Not publicly distributed here; per-request assessment under applicable agreements |
| CACTUS, A4C subset (38) | Dataset providers via the [CACTUS publication](https://doi.org/10.1016/j.compbiomed.2025.110003) and its data-access statement | Exact provider release and A4C file list/checksums; author-run version not recorded | Not publicly distributed here; per-request assessment under applicable agreements |
| EchoCP (60) | Dataset providers via the [EchoCP publication](https://doi.org/10.1007/978-3-030-87231-1_49); the collection inventory notes Kaggle access | Exact provider/Kaggle record, owner, version, and file checksums; author-run version not recorded | Not publicly distributed here; per-request assessment under applicable agreements |
| CardiacUDC (247) | Dataset providers via [GraphEcho](https://arxiv.org/abs/2309.11145); the collection inventory notes Kaggle access | Exact provider/Kaggle record, owner, version, and file checksums; author-run version not recorded | Not publicly distributed here; per-request assessment under applicable agreements |
| HMC-QU (149) | Contact the dataset authors via the [multi-view echocardiography publication](https://doi.org/10.1016/j.bspc.2023.105448) | Exact HMC-QU release/view subset and file checksums; author-run version not recorded | Not publicly distributed here; requires provider and institutional assessment |
| Private institutional subset (50) | EchoROI authors and the originating institution; no public acquisition route | Institution-approved internal subset/version, shared only through authorised channels | No public distribution promised; both source images and annotations require institutional permission |

The private subset may not be available to external researchers at all. The
complete original training membership is unresolved and the private subset may
be unavailable, so retraining does not guarantee the identical checkpoint or
reported metrics.
Consent for the original study does not establish permission for public
redistribution. A run using only accessible sources is a new experiment and
must report its changed composition; it cannot be claimed to reproduce the
original checkpoint.

The separately published MIMIC external-dataset contribution is not a blanket
release of this project's mixed-source scan-sector collection.

## If a package can be authorised

Supply annotation-only files, an access-specific manifest, decoding recipes,
and checksums following [the matching convention](../annotations/README.md).
Do not include source frames, cine files, DICOM metadata, or embedded LabelMe
`imageData`. Keep restricted source identifiers and file lists inside the
authorised package rather than the public repository.

The public [manifest example](../annotations/manifest.example.csv) is entirely
synthetic and grants no access. The local validator checks matching integrity,
not data permissions, de-identification, or clinical validity.
