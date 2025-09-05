#!/bin/bash
# Script to serve MkDocs documentation with correct Python path

cd "$(dirname "$0")"
export PYTHONPATH="backend:$PYTHONPATH"
backend/.venv/bin/mkdocs serve --dev-addr=127.0.0.1:8000
