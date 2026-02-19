# Spatial Sentinel: Predictive NCI & Somatic Tax Engine

## Project Overview
**Spatial Sentinel** is an urban analytics engine designed for users of advanced mobility aids (like Rollz Motion). The system predicts **Narrowing City Index (NCI)**—a metric representing the loss of accessible pedestrian space due to urban clutter (waste accumulation).

The core of the project is a transition from static mapping to **Probabilistic Spatial Analysis**, predicting where "Somatic Bottlenecks" (physical stress zones) will occur before the user encounters them.

---

## Technical Rigor & Data Architecture

### 1. Data Sources (The "Ground Truth")
The system rejects simulated data in favor of a multi-layered real-world data stack:
* **Infrastructure (GIGO):** Real-time extraction of waste disposal points from **OpenStreetMap (OSM)** via the `OSMnx` geometry engine.
* **Sidewalk Geometry:** High-fidelity sidewalk polygons extracted from **BGT (Basisregistratie Grootschalige Topografie)** in the **EPSG:28992 (RD New)** metric system.
* **Coordinate Fusion:** System-level transformation between metric BGT data and WGS84 GPS coordinates for real-time visualization.

### 2. The Predictive Likelihood Algorithm
Unlike standard object detection (which is reactive), Spatial Sentinel uses a **Temporal Weighting Function**:
$$L = (G \times W_{day})$$
* **G:** Static GIGO infrastructure coordinates.
* **W_day:** A temporal multiplier based on Amsterdam's waste collection cycles. 
    * *Current Calibration:* **Thursday = 1.0 (Peak Likelihood)** due to bulky waste collection in Science Park (Amsterdam Oost).

### 3. Somatic Tax Correlation
The system maps spatial "Risk Zones" (Heatmaps) that correlate with expected physiological stress.
* **Red Zones (7m radius):** Areas where the predicted **Residual Path Width (RPW)** falls below the safety threshold for mobility aids.
* **Predicted Impact:** Navigation through these zones is estimated to increase user heart rate (HR) by **15-20 BPM** compared to "Clear" segments.

---

## System Components & Tech Stack
* **Python 3.12:** Core processing logic.
* **GeoPandas & Shapely:** Spatial operations (buffering, unioning, and difference calculations).
* **OSMnx:** Real-world street network and infrastructure extraction.
* **Folium (Leaflet.js):** Interactive UI for visualizing the NCI Forecast.

---

## Implementation Roadmap
- [x] **Data Ingestion:** Successful integration of OSM and BGT data layers.
- [x] **Predictive Engine:** Implementation of day-of-week risk multipliers.
- [x] **Interactive Visualization:** Deployment of the dark-matter NCI Forecast Map.
- [ ] **Sensor Integration:** Live BLE synchronization with Shimmer GSR/HR sensors for real-time Somatic Tax validation.

---
*Developed for the SCDW_5 Framework | Science Park 904, Amsterdam*
