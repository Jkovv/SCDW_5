# Spatial Sentinel - Narrowing City Index

Brief project scaffold: a lightweight Python prediction engine to detect sidewalk bottlenecks in Amsterdam by
fusing static 3D city data with real-time informal barrier detections.

Contents:
- `engine.py`: TPW/RPW implementations, CityJSON ingest stubs (cjio), and a lightweight sensor listener for Raspberry Pi 500.
- `requirements.txt`: Python dependencies.

Quick local run (demo):
```bash
python3 engine.py
```

Example: run with real data (provide a CityJSON URL, Pi500 feed URL and homography file):
```bash
python3 engine.py --cityjson-url https://example.org/amsterdam_bgt.cityjson --sensor-url http://pi500.local:8080/detections --homography homography.npy --output bottlenecks.geojson
```

Notes:
- Before running on a Raspberry Pi 500, install dependencies from `requirements.txt`.
- Provide a calibrated homography matrix to map camera pixel coordinates to RD New (EPSG:28992). Accepts `.npy` or `.json` with 9 numbers.
- The `engine.py` file contains example usage and synthetic demonstrations of TPW/RPW. Output `bottlenecks.geojson` can be opened in GIS or converted to CityJSON for 3D Amsterdam viewer.
