@echo off
rem Start Riverwatch for a demo on Windows: double-click demo.cmd
rem
rem Sets up backend\.venv on the first run, adds synthetic demo reports if the
rem database is empty, and serves the dashboard at http://127.0.0.1:8000/.
rem Uses backend\models\model.pth (or PWW_MODEL_URL) when present; otherwise the
rem mock detector. Set PWW_DATABASE_URL first to use Tiger Data.
setlocal
cd /d "%~dp0backend"

if not exist ".venv\Scripts\python.exe" (
  echo Setting up Python ^(first run only^)...
  python -m venv .venv || goto :failed
)
".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r requirements.txt || goto :failed

if exist "models\model.pth" if "%PWW_MODEL_URL%"=="" (
  ".venv\Scripts\python.exe" -c "import torch, torchvision" 2>nul || (
    echo Found models\model.pth. Installing PyTorch to run it ^(first run only, a large download^)...
    ".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check torch torchvision || goto :failed
  )
)

if "%PWW_DETECTOR%"=="" if "%PWW_MODEL_URL%"=="" if not exist "models\model.pth" (
  echo No model found in backend\models\model.pth: using fake ^(mock^) detections.
  set PWW_DETECTOR=mock
)

".venv\Scripts\python.exe" -m app.seed --if-empty || goto :failed

echo.
echo Riverwatch is starting. Open http://127.0.0.1:8000/ in your browser.
echo Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
echo.
echo Riverwatch stopped.
pause
goto :eof

:failed
echo.
echo Something went wrong. Copy the messages above and send them to your team.
pause
