#!/usr/bin/env bash
# Start Riverwatch for a demo on macOS or Linux: ./demo.sh
#
# Sets up backend/.venv on the first run, adds synthetic demo reports if the
# database is empty, and serves the dashboard at http://127.0.0.1:8000/.
# Uses backend/models/model.pth (or PWW_MODEL_URL) when present; otherwise the
# mock detector. Set PWW_DATABASE_URL first to use Tiger Data.
set -euo pipefail
cd "$(dirname "$0")/backend"

if [ ! -x .venv/bin/python ]; then
  echo "Setting up Python (first run only)..."
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements.txt

if [ -f models/model.pth ] && [ -z "${PWW_MODEL_URL:-}" ] \
    && ! .venv/bin/python -c "import torch, torchvision" 2>/dev/null; then
  echo "Found models/model.pth. Installing PyTorch to run it (first run only, a large download)..."
  .venv/bin/python -m pip install --quiet --disable-pip-version-check torch torchvision
fi

if [ -z "${PWW_DETECTOR:-}" ] && [ -z "${PWW_MODEL_URL:-}" ] && [ ! -f models/model.pth ]; then
  echo "No model found in backend/models/model.pth: using fake (mock) detections."
  export PWW_DETECTOR=mock
fi

.venv/bin/python -m app.seed --if-empty

echo
echo "Riverwatch is starting. Open http://127.0.0.1:8000/ in your browser."
echo "Press Ctrl+C to stop."
echo
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
