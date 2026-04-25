# Specialist Artifacts

This directory keeps only the versioned metadata needed to discover, download,
and validate canonical specialists:

- `model_registry.json`
- `artifacts.manifest.json`

The binary assets themselves are intentionally not committed to Git:

- `*_canonical.pt`
- `*_preprocessing/`
- release ZIP bundles built from preprocessing artifacts

## Publish flow

1. Prepare release assets and refresh the manifest:

```bash
python scripts/prepare_specialist_release.py
```

2. Upload the generated files from `artifacts_archive/specialist_release_assets/`
   to a GitHub Release in this repository.

3. Commit the updated metadata files:

- `checkpoints/model_registry.json`
- `checkpoints/artifacts.manifest.json`

## Consumer flow

New developers should install the published specialists with:

```bash
python scripts/fetch_specialists.py --all
```

That command downloads the release assets, validates their `sha256`, extracts
preprocessing artifacts into `checkpoints/`, and leaves the registry-consistent
local layout expected by `python -m inference`.
