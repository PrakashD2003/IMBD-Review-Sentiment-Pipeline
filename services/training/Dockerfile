# ---------- Build stage ----------
FROM python:3.12-slim AS build
WORKDIR /app

# Faster, smaller builds
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build tools only here (discarded later)
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install deps into isolated venv (so runtime stays slim)
COPY requirements-training.txt .
RUN python -m venv /opt/venv \
  && /opt/venv/bin/pip install --no-cache-dir -r requirements-training.txt \
  && /opt/venv/bin/pip install --no-cache-dir dvc[s3]

# Copy only source needed at runtime
COPY services/training /app/services/training
COPY common /app/common
COPY dvc.yaml dvc.lock /app/

# ---------- Runtime stage ----------
FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Bring in the venv and app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app /app

# Optional: drop privileges (recommended)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Create a tiny entrypoint that can init/config DVC from env at runtime
# Supported env:
#   DVC_REMOTE_URL (optional): e.g. s3://bucket/path
#   (Credentials via IAM/IRSA or env; no secrets baked in layers)

# Put entrypoint where appuser has write perms
RUN printf '%s\n' \
  '#!/usr/bin/env sh' \
  'set -e' \
  'cd /app' \
  '' \
  '# Ensure a DVC repo exists (no SCM needed)' \
  '[ -d ".dvc" ] || dvc init --no-scm' \
  '' \
  '# Configure/refresh default remote if provided' \
  'if [ -n "$DVC_REMOTE_URL" ]; then' \
  '  if dvc remote list | grep -q "^storage"; then' \
  '    dvc remote modify storage url "$DVC_REMOTE_URL" || true' \
  '  else' \
  '    dvc remote add -f -d storage "$DVC_REMOTE_URL" || true' \
  '  fi' \
  'fi' \
  '' \
  'exec "$@"' > /app/entrypoint.sh \
  && sed -i 's/\r$//' /app/entrypoint.sh \
  && chmod +x /app/entrypoint.sh

EXPOSE 8001
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "services.training.api_end_point.training_fastapi_app:app", "--host", "0.0.0.0", "--port", "8001"]
