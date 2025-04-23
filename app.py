import pandas as pd
import numpy as np
import folium
import json
from flask import Flask, jsonify, request, render_template, redirect, url_for
from packaging_predictor import recommend_packaging
from main import get_recommended_materials

app = Flask(__name__, static_folder='static')

# Load the redistribution centers data
df = pd.read_csv('redistribution_center.txt')

def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

def find_nearest_center(user_lat, user_lon, centers_df):
    distances = centers_df.apply(
        lambda row: haversine_distance(user_lat, user_lon, row['latitude'], row['longtitude']),
        axis=1
    )
    nearest_idx = distances.idxmin()
    return centers_df.iloc[nearest_idx]

def create_map(user_location, nearest_center):
    # Create a map centered on Dubai
    dubai_map = folium.Map(location=[25.2048, 55.2708], zoom_start=11)
    
    # Add markers for all redistribution centers
    for idx, center in df.iterrows():
        # Determine if this is the nearest center
        is_nearest = center['center_name'] == nearest_center['center_name']
        
        # Choose color and icon based on whether it's the nearest center
        color = 'red' if is_nearest else 'gray'
        icon_type = 'info-sign' if is_nearest else 'home'
        
        folium.Marker(
            location=[center['latitude'], center['longtitude']],
            popup=center['center_name'],
            icon=folium.Icon(color=color, icon=icon_type)
        ).add_to(dubai_map)
    
    # Add marker for user location
    folium.Marker(
        location=user_location,
        popup='Your Location',
        icon=folium.Icon(color='blue', icon='user')
    ).add_to(dubai_map)
    
    # Draw line between user and nearest center
    folium.PolyLine(
        locations=[
            user_location,
            [nearest_center['latitude'], nearest_center['longtitude']]
        ],
        weight=2,
        color='green',
        opacity=0.8
    ).add_to(dubai_map)
    
    return dubai_map

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/product_recommender')
def product_recommender():
    # Load recommended materials from JSON
    with open('packaging_materials_list.json', 'r') as f:
        recommended_materials = json.load(f)
    
    return render_template('product_recommender.html', recommended_materials=recommended_materials)

@app.route('/recommend_packaging', methods=['POST'])
def process_packaging_recommendation():
    # Get form data
    product_type = request.form.get('product_type')
    weight = float(request.form.get('weight'))
    fragile = request.form.get('fragile')
    temp_condition = request.form.get('temp_condition')
    humidity_level = request.form.get('humidity_level')
    
    # Call packaging recommendation function
    recommendation = recommend_packaging(product_type, weight, fragile, temp_condition, humidity_level)
    
    # Load recommended materials from JSON
    with open('packaging_materials_list.json', 'r') as f:
        recommended_materials = json.load(f)
    
    return render_template('product_recommender.html', 
                           recommendation=recommendation,
                           recommended_materials=recommended_materials)

@app.route('/find_center', methods=['GET', 'POST'])
def find_center():
    result = None
    map_html = None
    google_maps_url = None
    
    if request.method == 'POST':
        try:
            user_lat = float(request.form.get('lat'))
            user_lon = float(request.form.get('lon'))
            
            user_location = (user_lat, user_lon)
            nearest = find_nearest_center(user_lat, user_lon, df)
            
            # Create the map
            dubai_map = create_map(user_location, nearest)
            map_html = dubai_map._repr_html_()
            
            # Create Google Maps directions URL
            google_maps_url = f"https://www.google.com/maps/dir/{user_lat},{user_lon}/{nearest['latitude']},{nearest['longtitude']}"
            
            result = {
                'nearest_center': nearest['center_name'],
                'center_coordinates': [nearest['latitude'], nearest['longtitude']],
                'distance': haversine_distance(user_lat, user_lon, nearest['latitude'], nearest['longtitude'])
            }
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return render_template('redistribution_center.html', result=result, map_html=map_html, google_maps_url=google_maps_url)

if __name__ == '__main__':
    app.run(debug=True) 