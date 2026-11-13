#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ ! -x .venv312/bin/python ]]; then
  echo 'Create a Python 3.12 environment first: python3.12 -m venv .venv312'
  echo 'Then install: .venv312/bin/python -m pip install -r traffic_sign_app/requirements.txt'
  exit 1
fi
exec .venv312/bin/python -m uvicorn traffic_sign_app.backend.main:app --host 127.0.0.1 --port "${PORT:-8000}"
