from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import folium

app = Flask(__name__)

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
    return '''
    <h1>LogiCore Route Finder</h1>
    <form action="/find_center" method="get">
        <label>Latitude:</label>
        <input type="number" step="any" name="lat" required><br>
        <label>Longitude:</label>
        <input type="number" step="any" name="lon" required><br>
        <input type="submit" value="Find Nearest Center">
    </form>
    '''

@app.route('/find_center')
def find_center():
    try:
        user_lat = float(request.args.get('lat', 25.2048))
        user_lon = float(request.args.get('lon', 55.2708))
        
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
        
        return f'''
        <div style="font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #2c3e50;">Results:</h2>
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <p><strong>Nearest center:</strong> {result['nearest_center']}</p>
                <p><strong>Distance:</strong> {result['distance']:.2f} km</p>
            </div>
            
            <div style="margin: 20px 0; text-align: center;">
                <a href="{google_maps_url}" target="_blank" style="
                    background-color: #4CAF50;
                    color: white;
                    padding: 15px 30px;
                    text-decoration: none;
                    border-radius: 5px;
                    display: inline-block;
                    font-size: 16px;
                    font-weight: bold;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    transition: all 0.3s ease;
                ">
                    📍 Get Directions in Google Maps
                </a>
            </div>
            
            <div style="width:100%; height:600px; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                {map_html}
            </div>
            
            <div style="margin-top: 20px; text-align: center;">
                <a href="/" style="color: #4CAF50; text-decoration: none;">← Back to home</a>
            </div>
        </div>
        '''
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/find_center')
def api_find_center():
    try:
        user_lat = float(request.args.get('lat', 25.2048))
        user_lon = float(request.args.get('lon', 55.2708))
        
        nearest = find_nearest_center(user_lat, user_lon, df)
        
        return jsonify({
            'nearest_center': nearest['center_name'],
            'center_coordinates': [nearest['latitude'], nearest['longtitude']],
            'distance': haversine_distance(user_lat, user_lon, nearest['latitude'], nearest['longtitude'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)