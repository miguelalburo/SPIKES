@echo off
echo ──────────────────────────────────────────
echo   SPIKES CGM Dashboard — Starting up...
echo ──────────────────────────────────────────
echo.

:: Check Docker is installed
docker --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not installed or not running.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop/
    echo Then re-run this file.
    pause
    exit /b 1
)

echo [1/3] Building the app image (first run takes ~2 minutes)...
docker build -t spikes-app .

echo.
echo [2/3] Starting the app...
docker run --rm -p 8501:8501 spikes-app &

echo.
echo [3/3] Opening your browser...
timeout /t 3 /nobreak >nul
start http://localhost:8501

echo.
echo App is running at http://localhost:8501
echo Close this window to stop the app.
pause
