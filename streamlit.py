import streamlit as st
import pandas as pd
from packaging_predictor import qa_chain
import json
import folium
from streamlit_folium import folium_static
import numpy as np
from datetime import datetime
from langchain_community.llms import Ollama
from langchain.output_parsers.json import parse_json_markdown

# Page config
st.set_page_config(
    page_title="LogiCore - Smart Logistics Solutions",
    page_icon="📦",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stSelectbox>div>div>select {
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("LogiCore Navigation")
page = st.sidebar.radio("Go to", ["Product Material Recommender", "Route Finder"])

if page == "Product Material Recommender":
    st.title("📦 Product Material Recommender")
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Monthly Packaging Requirements")
        # Load and process weather data
        df = pd.read_csv("dubai_2024_monthly_weather.csv")
        current_month = datetime.now().strftime("%B")
        month_index = df[df['month'] == current_month].index[0]
        next_4_months = [(month_index + i) % 12 for i in range(4)]
        next_month_names = df.iloc[next_4_months]['month'].tolist()
        
        # Build weather context
        weather_context = ", ".join(
            [f"{row['month']}: {row['weather']}" for idx, row in df.iloc[next_4_months].iterrows()]
        )
        
        # Get packaging materials
        prompt = f"""
        List ONLY the total packaging materials required for the next 4 months: {', '.join(next_month_names)}.
        Consider the weather conditions for these months: {weather_context}.
        Return a JSON array of item names. Example: ["Poly Bubble Mailers", "Waterproof Tape", "Insulated Boxes"]
        Give ONLY JSON, AND ALWAYS JSON. NO explanations, NO categories, ONLY the array.
        """
        
        llm = Ollama(model="llama3:latest")
        response = llm.invoke(prompt)
        parsed_data = parse_json_markdown(response)
        
        # Display materials in a table
        materials_df = pd.DataFrame(parsed_data, columns=["Required Materials"])
        st.dataframe(materials_df, use_container_width=True)
    
    with col2:
        st.subheader("Product Packaging Predictor")
        
        # Input form
        with st.form("product_form"):
            product_type = st.text_input("Product Type")
            weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
            fragile = st.selectbox("Is the product fragile?", ["Yes", "No"])
            temp_condition = st.selectbox("Temperature Condition", ["Room Temperature", "Refrigerated", "Frozen"])
            humidity_level = st.selectbox("Humidity Level", ["Low", "Medium", "High"])
            
            submitted = st.form_submit_button("Get Packaging Recommendation")
            
            if submitted:
                question = (f"What packaging should be used for a {product_type} "
                          f"that weighs {weight}kg, is fragile: {fragile}, "
                          f"requires {temp_condition} temperature and "
                          f"{humidity_level} humidity level?")
                
                result = qa_chain.run(question)
                st.success(f"Recommended Packaging: {result}")

else:
    st.title("🗺️ Route Finder")
    
    # Load redistribution centers data
    df = pd.read_csv('redistribution_center.txt')
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371
        return c * r
    
    def find_nearest_center(user_lat, user_lon, centers_df):
        distances = centers_df.apply(
            lambda row: haversine_distance(user_lat, user_lon, row['latitude'], row['longtitude']),
            axis=1
        )
        nearest_idx = distances.idxmin()
        return centers_df.iloc[nearest_idx]
    
    # Input form
    with st.form("location_form"):
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=25.2048, format="%.6f")
        with col2:
            lon = st.number_input("Longitude", value=55.2708, format="%.6f")
        
        submitted = st.form_submit_button("Find Nearest Center")
        
        if submitted:
            nearest = find_nearest_center(lat, lon, df)
            distance = haversine_distance(lat, lon, nearest['latitude'], nearest['longtitude'])
            
            # Display results
            st.success(f"Nearest Center: {nearest['center_name']}")
            st.info(f"Distance: {distance:.2f} km")
            
            # Create and display map
            m = folium.Map(location=[25.2048, 55.2708], zoom_start=11)
            
            # Add all centers
            for idx, center in df.iterrows():
                is_nearest = center['center_name'] == nearest['center_name']
                color = 'red' if is_nearest else 'gray'
                icon_type = 'info-sign' if is_nearest else 'home'
                
                folium.Marker(
                    location=[center['latitude'], center['longtitude']],
                    popup=center['center_name'],
                    icon=folium.Icon(color=color, icon=icon_type)
                ).add_to(m)
            
            # Add user location
            folium.Marker(
                location=[lat, lon],
                popup='Your Location',
                icon=folium.Icon(color='blue', icon='user')
            ).add_to(m)
            
            # Draw route
            folium.PolyLine(
                locations=[[lat, lon], [nearest['latitude'], nearest['longtitude']]],
                weight=2,
                color='green',
                opacity=0.8
            ).add_to(m)
            
            folium_static(m)
            
            # Google Maps link
            google_maps_url = f"https://www.google.com/maps/dir/{lat},{lon}/{nearest['latitude']},{nearest['longtitude']}"
            st.markdown(f"[Get Directions in Google Maps]({google_maps_url})", unsafe_allow_html=True) 