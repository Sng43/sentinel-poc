#!/usr/bin/env bash
# Deploy Project Sentinel to a Hugging Face Docker Space in one command.
#
# Prereqs (once):
#   1. Create a Space at https://huggingface.co/new-space  (SDK: Docker, blank)
#   2. Have a HF access token with write scope: https://huggingface.co/settings/tokens
#      When git prompts for a password on push, paste the TOKEN (not your HF password).
#
# Usage:
#   deploy/push_to_hf.sh https://huggingface.co/spaces/<user>/<space-name>
#
# It clones the (empty) Space, copies in exactly what the image needs — including
# the git-ignored test.parquet — drops the HF-flavoured README, commits, and pushes.
# HF then builds the Dockerfile automatically; watch the Space's "Logs" tab.
set -euo pipefail

SPACE_URL="${1:?Usage: push_to_hf.sh <hf-space-git-url>}"
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"     # sentinel-poc/deploy
PS="$(cd "$DEPLOY_DIR/../project_sentinel" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "→ cloning Space into staging dir…"
git clone "$SPACE_URL" "$STAGE/space"
cd "$STAGE/space"

echo "→ copying curated app files…"
cp "$PS/Dockerfile" "$PS/.dockerignore" "$PS/pyproject.toml" "$PS/uv.lock" "$PS/bench_latency.py" .
cp "$DEPLOY_DIR/hf-space-README.md" README.md
rm -rf src backend models frontend data
cp -R "$PS/src" "$PS/backend" "$PS/models" .
mkdir -p data/processed
# Slim the (full-data, ~63 MB) test set to a demo-sized sample before baking it into the
# image — the ward dashboard only ever shows ~12 patients, so a few hundred rows is plenty
# and keeps the Space image small + fast to boot. Seeded for reproducibility.
( cd "$PS" && uv run python -c "import pandas as pd; pd.read_parquet('data/processed/test.parquet').sample(n=800, random_state=42).to_parquet('$STAGE/space/data/processed/test.parquet', index=False)" )
echo "  → slimmed test.parquet to 800 rows for the demo"
# frontend source only (node_modules/dist are rebuilt inside the image)
mkdir -p frontend
rsync -a --exclude node_modules --exclude dist "$PS/frontend/" frontend/

echo "→ committing and pushing…"
git add -A
git commit -m "Deploy Project Sentinel demo" || { echo "nothing to commit"; exit 0; }
git push

echo "✓ pushed. HF is now building the Docker Space — open the Space and watch 'Logs'."
