# SPIKES CGM Dashboard — Setup Guide

## What you need (one-time install)

**Install Docker Desktop:**
- **Windows:** https://www.docker.com/products/docker-desktop/
- **Mac:** https://www.docker.com/products/docker-desktop/

After installing, open Docker Desktop and wait for it to say **"Docker is running"** in the bottom-left corner. You don't need to create an account.

---

## How to run the app

### Windows
1. Double-click **`run.bat`**
2. A black window will open — wait for it to finish (first time takes ~2 minutes)
3. Your browser will open automatically at `http://localhost:8501`

### Mac
1. Right-click **`run.sh`** → Open With → Terminal
   *(Or open Terminal, drag the file in, and press Enter)*
2. Wait for it to finish (first time takes ~2 minutes)
3. Your browser will open automatically at `http://localhost:8501`

---

## After the first run

The first run downloads and builds everything (~2 minutes). After that, starting the app takes about **10 seconds**.

---

## Stopping the app

- **Windows:** Close the black Command Prompt window
- **Mac:** Press `Ctrl+C` in the Terminal window

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Docker is not installed" | Install Docker Desktop and make sure it's open and running |
| Browser doesn't open automatically | Go to `http://localhost:8501` manually |
| Port already in use | Restart Docker Desktop and try again |
| App won't build | Make sure all `.py` files are in the same folder as `Dockerfile` |

---

## Files required in the same folder

```
Dockerfile
requirements.txt
SPIKES_app.py
spike_analysis.py
pillar_analysis.py
cohort_level.py
run.bat          ← Windows
run.sh           ← Mac
```
