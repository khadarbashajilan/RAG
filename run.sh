#!/usr/bin/env bash
# Launch the CLI. Env validation lives in stoic_rag.config so there is one
# source of truth; this script only sets up the environment and clears memory.
set -euo pipefail

cd "$(dirname "$0")"

rm -f data/checkpoints.db

if command -v uv >/dev/null 2>&1; then
    if [ ! -d .venv ]; then
        echo "Creating virtualenv..."
        uv sync
    fi
    exec uv run stoic-rag "$@"
fi

if [ -x .venv/bin/python ]; then
    exec .venv/bin/python -m stoic_rag.cli "$@"
fi

echo "No runner found."
echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
echo "or create a virtualenv with: python3 -m venv .venv"
exit 1
