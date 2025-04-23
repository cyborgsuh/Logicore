import pandas as pd

def get_unique_product_types(csv_file_path):
    try:
        df = pd.read_csv(csv_path)
        if 'Product_Type' not in df.columns:
            raise ValueError("Column 'product_type' not found in the CSV.")
        unique_values = df['Product_Type'].dropna().unique()
        return list(unique_values)
    except Exception as e:
        return f"Error: {e}"

# Example usage
csv_path = 'Product_Package_Dataset.csv'  # Replace with your actual file path
print(get_unique_product_types(csv_path))
