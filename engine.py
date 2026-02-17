# current engine file
"""
engine.py

Lightweight Spatial Sentinel scaffold implementing:
- TPW (Theoretical Path Width) via directional line sampling
- RPW (Residual Path Width) by subtracting obstacle footprints
- stubs for CityJSON ingest (cjio) and sensor JSON listener

Designed to be Raspberry Pi-friendly (avoid heavy pre-computation).
"""
from typing import List, Tuple, Optional
import time
import threading
import math
import argparse
import tempfile
import os
import json

import numpy as np

try:
	import geopandas as gpd
	from shapely.geometry import Point, LineString, Polygon, box
except Exception:
	gpd = None
	from shapely.geometry import Point, LineString, Polygon, box  # type: ignore

try:
	import requests
except Exception:
	requests = None

try:
	from sklearn.ensemble import RandomForestRegressor
	import joblib
except Exception:
	RandomForestRegressor = None
	joblib = None


def sample_cross_section(poly: Polygon, pt: Tuple[float, float], angles: int = 18) -> Tuple[float, Tuple[Tuple[float, float], Tuple[float, float]], float]:
	"""Compute TPW at point `pt` inside `poly` by sampling line directions.

	Returns: (best_length, (p0, p1), best_angle_radians)

	Approach: for `angles` samples over [0, pi), build a long line through `pt` and
	intersect with `poly`, take intersection segment length. Choose the maximum.
	This is a lightweight approximate method suitable for edge devices.
	"""
	if not poly.contains(Point(pt)):
		# if point is outside, return 0
		return 0.0, ((pt[0], pt[1]), (pt[0], pt[1])), 0.0

	best_len = 0.0
	best_seg = ((pt[0], pt[1]), (pt[0], pt[1]))
	best_angle = 0.0

	max_extent = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) * 2.0 + 10.0

	for i in range(angles):
		ang = (math.pi * i) / angles
		dx = math.cos(ang)
		dy = math.sin(ang)
		line = LineString([(pt[0] - dx * max_extent, pt[1] - dy * max_extent), (pt[0] + dx * max_extent, pt[1] + dy * max_extent)])
		inter = poly.intersection(line)
		# intersection may be MultiLineString, LineString, or empty
		seg_len = 0.0
		seg_coords = ((pt[0], pt[1]), (pt[0], pt[1]))
		if inter.is_empty:
			seg_len = 0.0
		else:
			# convert to longest line length inside poly along that direction
			if inter.geom_type == 'LineString':
				seg_len = inter.length
				seg_coords = (tuple(inter.coords)[0], tuple(inter.coords)[-1])
			else:
				# MultiLineString or GeometryCollection: pick longest
				try:
					parts = [g for g in inter]
					lengths = [p.length for p in parts if hasattr(p, 'length')]
					if lengths:
						idx = int(np.argmax(lengths))
						chosen = parts[idx]
						seg_len = chosen.length
						seg_coords = (tuple(chosen.coords)[0], tuple(chosen.coords)[-1])
				except Exception:
					seg_len = 0.0

		if seg_len > best_len:
			best_len = seg_len
			best_seg = seg_coords
			best_angle = ang

	return best_len, best_seg, best_angle


def compute_rpw_from_segment(segment: Tuple[Tuple[float, float], Tuple[float, float]], obstacles: List[Polygon]) -> float:
	"""Given a segment (two endpoints) representing the best cross-section and a list of obstacle footprints,
	compute the largest continuous free subsegment length after subtracting obstacle projections.
	"""
	seg_line = LineString([segment[0], segment[1]])
	free = seg_line
	# subtract all obstacle intersections along the line
	# each obstacle may clip the line to segments; we compute difference
	for obs in obstacles:
		inter = free.intersection(obs)
		if not inter.is_empty:
			free = free.difference(obs)
			# difference may become MultiLineString or LineString
			if free.is_empty:
				return 0.0

	# now compute the longest remaining continuous piece
	max_len = 0.0
	if free.geom_type == 'LineString':
		max_len = free.length
	else:
		try:
			parts = [g for g in free]
			lengths = [p.length for p in parts if hasattr(p, 'length')]
			if lengths:
				max_len = float(np.max(lengths))
		except Exception:
			max_len = 0.0

	return max_len


def project_pixel_to_rd(homography: np.ndarray, pixel: Tuple[float, float], depth: float) -> Tuple[float, float]:
	"""Project a camera pixel + depth to RD New coordinates using a homography.

	Note: This is a simplified example. In practice use calibrated camera intrinsics, extrinsics and
	a full pinhole+pose model. Here we assume the homography encodes mapping from grounded plane.
	"""
	u, v = pixel
	vec = np.array([u, v, 1.0])
	mapped = homography.dot(vec)
	if mapped[2] == 0:
		raise ValueError("Invalid homography mapping")
	x = mapped[0] / mapped[2]
	y = mapped[1] / mapped[2]
	# If depth scaling required, apply small correction (user must calibrate)
	return float(x), float(y)


def start_sensor_listener(url: str, callback, poll_interval: float = 0.5):
	"""Simple polling listener for the Pi500 JSON feed. Calls `callback(data)` on new data.

	Designed as a lightweight thread loop for Pi.
	"""
	if requests is None:
		raise RuntimeError("requests not available")

	stop_flag = threading.Event()

	def run():
		while not stop_flag.is_set():
			try:
				r = requests.get(url, timeout=2.0)
				if r.status_code == 200:
					j = r.json()
					callback(j)
			except Exception:
				pass
			time.sleep(poll_interval)

	t = threading.Thread(target=run, daemon=True)
	t.start()
	return stop_flag


def download_file(url: str, dst: Optional[str] = None, timeout: float = 10.0) -> Optional[str]:
	"""Download a URL to local file and return path. Returns None on failure."""
	if requests is None:
		return None
	try:
		r = requests.get(url, stream=True, timeout=timeout)
		if r.status_code != 200:
			return None
		if dst is None:
			fd, dst = tempfile.mkstemp(suffix='.json')
			os.close(fd)
		with open(dst, 'wb') as f:
			for chunk in r.iter_content(chunk_size=8192):
				if chunk:
					f.write(chunk)
		return dst
	except Exception:
		return None


def load_homography(path: str) -> Optional[np.ndarray]:
	"""Load a 3x3 homography from a JSON or NumPy file.

	Accepts .npy (numpy array) or .json with a flat list of 9 values.
	"""
	if path.endswith('.npy'):
		try:
			return np.load(path)
		except Exception:
			return None
	try:
		with open(path, 'r') as f:
			j = json.load(f)
			arr = np.array(j)
			if arr.size == 9:
				return arr.reshape((3,3))
	except Exception:
		pass
	return None


def detections_to_obstacles(detections: List[dict], homography: np.ndarray, depth_scale: float = 1.0) -> List[Polygon]:
	"""Convert detection dicts into ground-plane obstacle polygons in RD coordinates.

	Expected detection dict format (example):
	  {'bbox': [x1,y1,x2,y2], 'depth': 1.2, 'class': 'trash'}

	Approach: map bottom-left and bottom-right pixel coords through homography to RD.
	Create a narrow polygon by buffering the segment by an estimated half-width.
	"""
	obstacles: List[Polygon] = []
	for det in detections:
		try:
			bbox = det.get('bbox', None)
			if bbox is None or len(bbox) != 4:
				continue
			x1, y1, x2, y2 = bbox
			# bottom-left, bottom-right in pixel space (image origin top-left)
			bl = (x1, y2)
			br = (x2, y2)
			p_bl = project_pixel_to_rd(homography, bl, det.get('depth', 1.0))
			p_br = project_pixel_to_rd(homography, br, det.get('depth', 1.0))
			seg = LineString([p_bl, p_br])
			depth = float(det.get('depth', 1.0)) * depth_scale
			# estimate half-width (meters): use bbox pixel height mapped to meters via depth proxy
			pixel_height = max(1.0, abs(y2 - y1))
			half_w = max(0.2, min(1.5, depth * 0.25))
			poly = seg.buffer(half_w, cap_style=2)
			obstacles.append(poly)
		except Exception:
			continue
	return obstacles


def process_sensor_payload(payload: dict, sidewalks_gdf, homography: np.ndarray, out_geojson_path: str = 'bottlenecks.geojson'):
	"""Process a JSON payload from the Pi500: project detections, compute RPW and emit alerts/features.

	sidewalks_gdf: GeoDataFrame in EPSG:28992 containing sidewalk polygons.
	"""
	features = []
	dets = payload.get('detections') or payload.get('objects') or []
	obstacles = detections_to_obstacles(dets, homography)

	# For each obstacle, find nearest sidewalk polygon and compute TPW/RPW
	if sidewalks_gdf is None or sidewalks_gdf.empty:
		return

	for obs in obstacles:
		centroid = obs.centroid
		# spatial index usage if available
		candidates = sidewalks_gdf
		try:
			if hasattr(sidewalks_gdf, 'sindex') and sidewalks_gdf.sindex is not None:
				idx = list(sidewalks_gdf.sindex.nearest((centroid.x, centroid.y, centroid.x, centroid.y), 1))
				candidates = sidewalks_gdf.iloc[idx]
		except Exception:
			candidates = sidewalks_gdf

		for _, row in candidates.iterrows():
			poly = row.geometry
			if not poly.contains(centroid):
				continue
			# choose a point inside poly to evaluate TPW (centroid)
			pt = (centroid.x, centroid.y)
			tpw, seg, ang = sample_cross_section(poly, pt, angles=24)
			rpw = compute_rpw_from_segment(seg, [obs])
			props = {
				'tpw': float(tpw),
				'rpw': float(rpw),
				'alert': bool(rpw < 0.9),
			}
			geom = obs.__geo_interface__
			feat = {'type':'Feature','geometry':geom,'properties':props}
			features.append(feat)
			if props['alert']:
				print(f"Bottleneck ALERT: RPW={rpw:.2f}m at {pt}")

	if features:
		geo = generate_bottleneck_geojson(features)
		try:
			with open(out_geojson_path, 'w') as f:
				json.dump(geo, f)
			print(f"Wrote {len(features)} features to {out_geojson_path}")
		except Exception:
			pass


def ingest_cityjson_sidewalks(path: str):
	"""Attempt to ingest sidewalks from a CityJSON using `cjio`.

	Returns a GeoDataFrame of sidewalk polygons in EPSG:28992 if possible.
	This function is tolerant: if `cjio` is not installed or file not available, it returns None.
	"""
	# If the input is a GeoJSON/GeoPackage, prefer GeoPandas read
	if gpd is not None:
		try:
			if path.lower().endswith(('.geojson', '.json', '.gpkg', '.shp')):
				gdf = gpd.read_file(path)
				if gdf is not None and not gdf.empty:
					# Ensure CRS is set to RD New for downstream math (user should provide correct CRS)
					if gdf.crs is None:
						gdf.set_crs('EPSG:28992', inplace=True)
					else:
						gdf = gdf.to_crs('EPSG:28992')
					return gdf
		except Exception:
			pass

	# Fallback: try CityJSON via cjio
	try:
		import cjio
	except Exception:
		return None

	# Minimal example: load and extract features with attribute identifying sidewalks (implementation depends on dataset)
	cj = cjio.CityJSON(path)
	# user must adapt the filter below to the actual cityjson schema (BGT specifics)
	polys = []
	for fid, feat in cj.cityobjects.items():
		if feat.get('type', '').lower() in ('sidewalk', 'straatmeubilair', 'wegdeel'):
			geom = cj.get_feature_geometry(fid)
			try:
				poly = Polygon(geom[0][0])
				polys.append(poly)
			except Exception:
				continue

	if not polys:
		return None

	if gpd is None:
		return None

	gdf = gpd.GeoDataFrame({'geometry': polys}, crs='EPSG:28992')
	return gdf


def train_random_forest(X, y):
	if RandomForestRegressor is None:
		raise RuntimeError('scikit-learn not available')
	rf = RandomForestRegressor(n_estimators=50, random_state=42)
	rf.fit(X, y)
	return rf


def generate_bottleneck_geojson(features: List[dict]) -> dict:
	"""Return a GeoJSON FeatureCollection for bottlenecks. Each feature must include 'geometry' and 'properties'."""
	return {
		"type": "FeatureCollection",
		"features": features
	}


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Spatial Sentinel engine CLI')
	parser.add_argument('--cityjson-url', required=True, help='URL to CityJSON (BGT) file or local path')
	parser.add_argument('--sensor-url', required=True, help='URL for Pi500 JSON feed')
	parser.add_argument('--homography', required=True, help='Path to homography (.npy or .json)')
	parser.add_argument('--output', help='Output GeoJSON path', default='bottlenecks.geojson')
	args = parser.parse_args()

	# Download/load CityJSON
	src = args.cityjson_url
	path = None
	if src.startswith('http://') or src.startswith('https://'):
		print('Downloading CityJSON...')
		path = download_file(src)
	else:
		path = src if os.path.exists(src) else None

	if not path:
		print('Error: CityJSON path not available or download failed')
		raise SystemExit(2)

	sidewalks = ingest_cityjson_sidewalks(path)
	if sidewalks is None or sidewalks.empty:
		print('Error: failed to extract sidewalks from CityJSON')
		raise SystemExit(3)
	print(f'Loaded {len(sidewalks)} sidewalk polygons')

	# Load homography
	hom = load_homography(args.homography)
	if hom is None:
		print('Error: failed to load homography file')
		raise SystemExit(4)

	# Start sensor listener and require live data
	def cb(payload):
		try:
			process_sensor_payload(payload, sidewalks, hom, out_geojson_path=args.output)
		except Exception as e:
			print('Error processing payload:', e)

	print('Starting sensor listener (live mode)...')
	stop = start_sensor_listener(args.sensor_url, cb)
	try:
		while True:
			time.sleep(1.0)
	except KeyboardInterrupt:
		stop.set()

