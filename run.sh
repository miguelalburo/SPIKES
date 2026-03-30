#!/bin/bash
echo "──────────────────────────────────────────"
echo "  SPIKES CGM Dashboard — Starting up..."
echo "──────────────────────────────────────────"
echo ""

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not running."
    echo "Please install Docker Desktop from https://www.docker.com/products/docker-desktop/"
    echo "Then re-run this script."
    exit 1
fi

echo "[1/3] Building the app image (first run takes ~2 minutes)..."
docker build -t spikes-app .

echo ""
echo "[2/3] Starting the app..."
docker run --rm -p 8501:8501 spikes-app &

echo ""
echo "[3/3] Opening your browser..."
sleep 3
open http://localhost:8501 2>/dev/null || xdg-open http://localhost:8501 2>/dev/null || \
    echo "Please open http://localhost:8501 in your browser."

echo ""
echo "App is running at http://localhost:8501"
echo "Press Ctrl+C to stop the app."
wait
