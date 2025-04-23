import streamlit as st
import pandas as pd
from packaging_predictor import qa_chain
from datetime import datetime
from langchain_community.llms import Ollama
from langchain.output_parsers.json import parse_json_markdown
import json

# Page config
st.set_page_config(
    page_title="LogiCore - Packaging Solutions",
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
    
    # Save to JSON file
    filename = "packaging_materials_list.json"
    with open(filename, "w") as f:
        json.dump(parsed_data, f, indent=4)
    st.success(f"✅ JSON file saved as {filename}")

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