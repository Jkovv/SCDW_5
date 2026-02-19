import osmnx as ox
import folium
from folium.plugins import HeatMap
import geopandas as gpd
from shapely.ops import unary_union
from datetime import datetime
import os

def run_spatial_sentinel():
    print("INITIALIZING SPATIAL SENTINEL ENGINE...")
    
    # --- KROK 1: POBIERANIE PRAWDZIWYCH DANYCH ---
    # Lokalizacja: Science Park 904 (FNWI), Amsterdam
    location_point = (52.3558, 4.9555)
    dist = 500 
    
    print(f"Downloading real data for Science Park from OpenStreetMap...")
    
    try:
        # sidewalks
        sidewalks = ox.features_from_point(location_point, tags={"highway": "footway", "footway": "sidewalk"}, dist=dist)
        
        # gigo
        waste_bins = ox.features_from_point(location_point, tags={
            "amenity": ["waste_disposal", "waste_basket"],
            "bin": "yes"
        }, dist=dist)
        
        print(f"Data Ingested: {len(sidewalks)} sidewalk segments, {len(waste_bins)} waste points.")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    # pred 
    today = datetime.now().strftime('%A')
    
    # Definiujemy prawdopodobieństwo (Likelihood) blokady:
    # Czwartek = 1.0 (Szczyt), inne dni = 0.4
    risk_weight = 1.0 if today == "Thursday" else 0.4
    
    print(f"Today is {today}. Trash Likelihood Weight: {risk_weight * 100}%")

    # Przygotowanie danych do Heatmapy (lat, lon, weight)
    # Używamy centroidów, bo śmietniki w OSM mogą być punktami lub poligonami
    heat_data = []
    for _, row in waste_bins.iterrows():
        centroid = row.geometry.centroid
        heat_data.append([centroid.y, centroid.x, risk_weight])

    print("Generating final NCI Likelihood Map...")
    
    # map
    m = folium.Map(location=location_point, zoom_start=17, tiles='CartoDB dark_matter')

    # sidewalks
    folium.GeoJson(
        sidewalks.to_crs(epsg=4326),
        name="Real Sidewalks",
        style_function=lambda x: {'color': '#3498db', 'weight': 2, 'opacity': 0.6}
    ).add_to(m)

    # heatmap of likelihood (radius 30)
    HeatMap(heat_data, radius=30, blur=20, min_opacity=0.4).add_to(m)

    # GIGO 
    for idx, row in waste_bins.iterrows():
        centroid = row.geometry.centroid
        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=3,
            color='#e67e22',
            fill=True,
            popup=f"GIGO Infrastructure Point<br>Likelihood: {risk_weight*100}%"
        ).add_to(m)

    title_html = f'''
             <div style="position: fixed; 
                         bottom: 50px; left: 50px; width: 300px; height: 100px; 
                         background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                         padding: 10px; opacity: 0.85;">
             <b>Spatial Sentinel: NCI Forecast</b><br>
             <b>Date:</b> {datetime.now().strftime('%Y-%m-%d (%A)')}<br>
             <b>Status:</b> {'CRITICAL BLOCKAGE RISK' if risk_weight == 1.0 else 'NORMAL'}<br>
             <i style="color:red">Red Zones</i>: High probability of trash accumulation.
             </div>
             '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Zapis
    m.save("spatial_sentinel_report.html")
    print("\n Success! Open 'spatial_sentinel_report.html' to see your results.")

if __name__ == "__main__":
    run_spatial_sentinel()