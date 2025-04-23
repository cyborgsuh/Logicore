import pandas as pd
from datetime import datetime
from langchain_community.llms import Ollama
from langchain.output_parsers.json import parse_json_markdown
import json

def get_recommended_materials():
    """
    Get the list of recommended packaging materials for the next 4 months
    based on weather conditions.
    
    Returns:
        list: List of recommended packaging materials
    """
    try:
        # First try to load from the existing JSON file
        with open("packaging_materials_list.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # If file doesn't exist, generate new recommendations
        return generate_recommendations()

def generate_recommendations():
    """
    Generate fresh recommendations based on weather conditions
    """
    #Loading and getting the next 4 months
    df = pd.read_csv("dubai_2024_monthly_weather.csv")

    current_month = datetime.now().strftime("%B")
    month_index = df[df['month'] == current_month].index[0]
    next_4_months = [(month_index + i) % 12 for i in range(4)]
    next_month_names = df.iloc[next_4_months]['month'].tolist()

    # Build weather context for the prompt
    weather_context = ", ".join(
        [f"{row['month']}: {row['weather']}" for idx, row in df.iloc[next_4_months].iterrows()]
    )
    #Prompt
    prompt = f"""
    List ONLY the total packaging materials required for the next 4 months: {', '.join(next_month_names)}.
    Consider the weather conditions for these months: {weather_context}.
    Return a JSON array of item names. Example: ["Poly Bubble Mailers", "Waterproof Tape", "Insulated Boxes"]
    Give ONLY JSON, AND ALWAYS JSON. NO explanations, NO categories, ONLY the array.
    """

    llm = Ollama(model="llama3:latest")
    response = llm.invoke(prompt)
    parsed_data = parse_json_markdown(response)

    filename = "packaging_materials_list.json"
    with open(filename, "w") as f:
        json.dump(parsed_data, f, indent=4)

    print(f"✅ JSON file saved as {filename}")
    return parsed_data

if __name__ == "__main__":
    materials = generate_recommendations()
    print(materials)
