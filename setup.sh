#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"

# qwen-tts needs accelerate==1.12.0, which requires Python >=3.10
for py in python3.12 python3.11 python3.10 python3; do
  if command -v "$py" &>/dev/null && "$py" -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PYTHON="$py"
    break
  fi
done
if [[ -z "${PYTHON:-}" ]]; then
  echo "Error: Python 3.10+ is required (accelerate 1.12.0 does not support Python 3.9)."
  echo "Install with: brew install python@3.11  then use: python3.11 -m venv .venv"
  exit 1
fi
echo "Using: $(${PYTHON} --version)"

echo "Creating virtual environment: ${VENV_DIR}"
"${PYTHON}" -m venv "${VENV_DIR}"

echo "Installing Python dependencies..."
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r requirements.txt

echo "Setup complete."
echo "Activate and run with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python scripts/generate_tts.py"
