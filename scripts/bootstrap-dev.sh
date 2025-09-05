#!/usr/bin/env bash
set -euo pipefail

# This script bootstraps the monorepo for local development in PyCharm or terminal.
# It installs both Python subprojects in editable mode into the CURRENT interpreter.
#
# Usage:
#   1) Create/activate your virtualenv for the monorepo (recommended at repo root).
#   2) Run: bash scripts/bootstrap-dev.sh
#   3) In PyCharm: set the Project Interpreter to the same venv; open the repo root.
#
# Notes:
# - We intentionally avoid creating or activating a venv here to respect your environment.
# - Hatchling and setuptools both support PEP 660 editable installs (pip >= 21.3 recommended).

# Detect python/pip
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

PIP="${PY} -m pip"

# Ensure modern build tooling
${PIP} install --upgrade pip setuptools wheel

# Install SDK first (provides kei_agent_py_sdk)
${PIP} install -e ./kei-agent-py-sdk

# Install backend (provides keiko-backend package code)
${PIP} install -e ./backend

echo "\nBootstrap complete. Verify in PyCharm:"
echo "  - Settings > Project > Python Interpreter => select this venv"
echo "  - File > Invalidate Caches / Restart if inspections still show unresolved imports"
echo "  - Mark backend and kei-agent-py-sdk as Source Roots (Project Structure) if needed"
