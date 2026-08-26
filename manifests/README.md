# Artifact manifests

- `datasets/`: public metadata, provenance and full-file SHA-256 lists for private datasets.
- `models/act_models.csv`: logical checkpoint inventory and per-file SHA-256; model files remain private.
- `evaluations/`: rollout protocol metadata.

Dataset SHA-256 files use paths relative to the directory containing each dataset. Verify from the private artifact root with `sha256sum -c <manifest>` after adjusting only the leading storage directory if necessary.
