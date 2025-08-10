#!/usr/bin/env bash
set -euo pipefail

echo "[post-train] Pushing DVC-tracked data..."
dvc push

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
