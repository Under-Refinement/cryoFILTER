#!/usr/bin/env bash
# Run from any directory. All environment, cache, and temporary files stay here.
set -euo pipefail
setup=false
build=false
for arg in "$@"; do
  case "$arg" in
    --setup) setup=true ;;
    --build) build=true ;;
    *) echo "Usage: bash docsite/serve.sh [--setup] [--build]" >&2; exit 2 ;;
  esac
done
docsite_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$docsite_dir"
mkdir -p .tmp .cache .preview/config
export TMPDIR="$docsite_dir/.tmp"
export XDG_CACHE_HOME="$docsite_dir/.cache"
export XDG_CONFIG_HOME="$docsite_dir/.preview/config"

if [[ "$setup" == true ]]; then
  # The pinned packages support Python 3.10–3.14, not only Python 3.11.
  # Reuse an existing docs interpreter, then look for a usable local Python.
  docs_candidates=(.venv/bin/python python3 python3.11 python3.12 python3.13 python3.14 python3.10 python)
  if [[ -n "${DOCS_PYTHON:-}" ]]; then
    docs_candidates=("$DOCS_PYTHON")
  fi
  docs_python=""
  for candidate in "${docs_candidates[@]}"; do
    if "$candidate" -c 'import sys, venv, ensurepip; sys.exit(not ((3, 10) <= sys.version_info[:2] < (3, 15)))' >/dev/null 2>&1; then
      docs_python="$candidate"
      break
    fi
  done
  if [[ -z "$docs_python" ]]; then
    echo "Docs setup needs Python 3.10–3.14 with venv and ensurepip available." >&2
    echo "Activate an existing compatible Python environment and retry, or set DOCS_PYTHON to its interpreter." >&2
    exit 1
  fi
  echo "Setting up docs with $docs_python ($("$docs_python" --version 2>&1))."
  "$docs_python" -m venv .venv
  PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
    .venv/bin/python -m pip install --no-cache-dir \
    --index-url https://pypi.org/simple -r requirements.txt
fi
if [[ ! -x .venv/bin/mkdocs ]]; then
  echo "First run: bash docsite/serve.sh --setup --build (requires Python 3.10–3.14)." >&2
  exit 1
fi
if [[ "$build" == true ]]; then
  .venv/bin/mkdocs build --strict
  echo "Tutorial ready. Launch cryoFILTER app and open its Tutorial link."
  exit 0
fi
exec .venv/bin/mkdocs serve --dev-addr "127.0.0.1:${DOCS_PORT:-8000}"
