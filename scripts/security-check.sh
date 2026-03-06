#!/bin/bash
# Simple security check script for local development
# Usage: ./scripts/security-check.sh

set -e

echo "🔒 Running security checks..."

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Virtual environment not activated. Activating .venv..."
    source .venv/bin/activate 2>/dev/null || {
        echo "❌ Error: Could not activate virtual environment"
        echo "Please ensure you have a virtual environment at .venv"
        exit 1
    }
fi

# Install security tools if not present
echo "📦 Ensuring security tools are installed..."
uv pip install -q safety bandit

# Run Safety check for known vulnerabilities
echo ""
echo "🔍 Checking dependencies for known vulnerabilities..."
safety check || {
    echo "⚠️  Warning: Some dependencies have known vulnerabilities"
}

# Run Bandit for code security issues
echo ""
echo "🔍 Scanning code for security issues..."
bandit -r . --exclude ./.venv,./tests -ll || {
    echo "⚠️  Warning: Some security issues found in code"
}

echo ""
echo "✅ Security checks complete!"
echo ""
echo "💡 Tip: For more detailed reports, run:"
echo "   safety check --full-report"
echo "   bandit -r . --exclude ./.venv,./tests -v"