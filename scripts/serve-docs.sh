#!/bin/bash
set -e

# Script to serve the Keiko documentation including KEI-Agent-Framework
# Uses the existing uv virtual environment in backend/

echo "🚀 Starting Keiko Documentation Server (including KEI-Agent-Framework)..."

# Check if we're in the right directory
if [ ! -f "backend/pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Expected to find backend/pyproject.toml"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "backend/.venv" ]; then
    echo "📦 Virtual environment not found. Installing dependencies..."
    cd backend/
    uv sync --extra dev
    cd ..
else
    echo "✅ Using existing virtual environment in backend/.venv"
fi

# Check if mkdocs is available
if ! backend/.venv/bin/mkdocs --version > /dev/null 2>&1; then
    echo "❌ Error: mkdocs not found in virtual environment"
    echo "   Installing development dependencies..."
    cd backend/
    uv sync --extra dev
    cd ..
fi

# Navigate to docs directory
cd docs/

echo "📚 Serving documentation at http://localhost:8000"
echo "📖 KEI-Agent-Framework docs available at: http://localhost:8000/frameworks/kei-agent/docs/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the documentation server
../backend/.venv/bin/mkdocs serve
