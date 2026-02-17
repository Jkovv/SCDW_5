# Spatial Sentinel - Narrowing City Index (scaffold)

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

Local quick-check (mock sensor)
--------------------------------
You can mock a simple sensor feed locally and run the engine against local files to verify the pipeline. Example steps:

1. Create a simple homography (identity) file:
```bash
python3 -c "import numpy as np; np.save('homography.npy', np.eye(3))"
```

2. Create a mock detection file `detections.json` in a folder (example contents):
```json
{
	"detections": [
		{"bbox": [100, 50, 200, 150], "depth": 2.0, "class": "trash"}
	]
}
```

3. Serve that folder with a simple HTTP server (from the folder containing `detections.json`):
```bash
python3 -m http.server 8000
```

4. Run the engine pointing to your CityJSON file (or a downloaded CityJSON), the mock sensor URL and the homography. Example command:
```bash
python3 engine.py --cityjson-url /path/to/your_amsterdam_bgt.cityjson --sensor-url http://localhost:8000/detections.json --homography homography.npy --output test_bottlenecks.geojson
```

If all resources load, the engine will poll `http://localhost:8000/detections.json`, project detections using `homography.npy`, compute TPW/RPW against sidewalks extracted from the CityJSON, and write `test_bottlenecks.geojson` with any detected bottlenecks.