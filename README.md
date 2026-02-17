# Spatial Sentinel — Narrowing City Index (scaffold)

Brief project scaffold: a lightweight Python prediction engine to detect sidewalk bottlenecks in Amsterdam by
fusing static 3D city data with real-time informal barrier detections.

Contents:
- `engine.py`: TPW/RPW implementations, CityJSON ingest stubs (cjio), and a lightweight sensor listener for Raspberry Pi 500.
- `requirements.txt`: Python dependencies.

Quick local run (demo):
```bash
python3 engine.py
```

Notes:
- Before running on a Raspberry Pi 500, install dependencies from `requirements.txt`.
- Provide a calibrated homography matrix to map camera pixel coordinates to RD New (EPSG:28992).
- The `engine.py` file contains example usage and synthetic demonstrations of TPW/RPW.