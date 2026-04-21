Backend container build files live here so they are versioned with the backend code.

Use one of these tracked entry points instead of relying on the unversioned top-level
`/Users/rniyer/bowliverse_14/Dockerfile`:

Local build:

```bash
/Users/rniyer/bowliverse_14/app/tools/build_backend_image.sh gcr.io/<project>/<image>:<tag>
```

Cloud Build:

```bash
gcloud builds submit /Users/rniyer/bowliverse_14 \
  --config /Users/rniyer/bowliverse_14/app/cloudbuild.backend.yaml \
  --substitutions _IMAGE=gcr.io/<project>/<image>:<tag>
```

The tracked Docker build explicitly pre-fetches the heavy MediaPipe pose model during
image build so cold requests do not trigger a runtime model download.
