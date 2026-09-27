#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "Missing .env file."
    echo "Create one with:"
    echo "  GEMINI_API_KEY=your_gemini_api_key"
    echo "  PINECONE_API_KEY=your_pinecone_api_key"
    exit 1
fi

if ! grep -qE '^MISTRAL_API_KEY=.+$' .env; then
    echo "MISTRAL_API_KEY is not set in .env"
    exit 1
fi

if ! grep -qE '^GEMINI_API_KEY=.+$' .env; then
    echo "GEMINI_API_KEY is not set in .env"
    exit 1
fi

if ! grep -qE '^PINECONE_API_KEY=.+$' .env; then
    echo "PINECONE_API_KEY is not set in .env"
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    if [ ! -d .venv ]; then
        echo "Creating virtualenv..."
        uv sync
    fi
    rm -f checkpoints.db
    exec uv run main.py "$@"
fi

if [ -x .venv/bin/python ]; then
    exec .venv/bin/python main.py "$@"
fi

echo "No runner found."
echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
echo "or create a virtualenv with: python3 -m venv .venv"
exit 1
