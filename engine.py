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


def ingest_cityjson_sidewalks(path: str):
	"""Attempt to ingest sidewalks from a CityJSON using `cjio`.

	Returns a GeoDataFrame of sidewalk polygons in EPSG:28992 if possible.
	This function is tolerant: if `cjio` is not installed or file not available, it returns None.
	"""
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
	# quick local demo of TPW/RPW on a synthetic sidewalk polygon and an obstacle
	from shapely.geometry import Polygon

	# synthetic sidewalk (simple rectangle)
	sidewalk = Polygon([(0,0),(10,0),(10,3),(0,3)])
	pt = (5,1.5)
	tpw, seg, angle = sample_cross_section(sidewalk, pt, angles=36)
	print(f"TPW approx: {tpw:.2f} m, segment: {seg}, angle: {angle:.2f} rad")

	# synthetic obstacle occupying right half of the segment
	obs = Polygon([(6,0.5),(10,0.5),(10,2.5),(6,2.5)])
	rpw = compute_rpw_from_segment(seg, [obs])
	print(f"RPW after obstacle: {rpw:.2f} m")

