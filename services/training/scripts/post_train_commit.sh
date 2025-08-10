#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=$(command -v python || command -v python3 || command -v py || true)

echo "[post-train] Pushing DVC-tracked data..."
if command -v dvc >/dev/null 2>&1; then
  dvc push
else
  if [[ -z "${PYTHON_BIN}" ]]; then
    echo "[post-train] Python interpreter not found. Cannot run 'dvc push'." >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m dvc push
fi

echo "[post-train] Staging pipeline files..."
git add -A dvc.lock dvc.yaml 2>/dev/null || true

# Commit only if there’s something staged
if git diff --cached --quiet; then
  echo "[post-train] No changes to commit."
else
  timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
  git commit -m "Post-training commit ${timestamp}"
fi

# Push only if there’s an upstream branch
branch=$(git rev-parse --abbrev-ref HEAD)
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  echo "[post-train] Pushing commit to origin/${branch}..."
  git push origin "${branch}"
else
  echo "[post-train] No upstream set for branch ${branch}. Skipping push."
fi

echo "[post-train] Done."
