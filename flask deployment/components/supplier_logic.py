import pandas as pd
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster
import os

print("Starting supplier_logic.py") # Added

# Load data (ensure the path is correct relative to this file)
try:
    print("Attempting to load CSV file...") # Added
    # Get the absolute path to the CSV file for debugging
    csv_path = os.path.abspath('utils\df_fournisseurs.csv')
    print(f"Absolute path to CSV: {csv_path}") # Added
    df_fournisseurs = pd.read_csv(csv_path, encoding='utf-8-sig')
    print("CSV file loaded successfully.") # Added
    print(df_fournisseurs.head())  # Print the first few rows for inspection
    df_fournisseurs.columns = df_fournisseurs.columns.str.strip()
    df_fournisseurs = df_fournisseurs.rename(columns={
        'Société': 'Societe',
        'Biostimulants & Growth Enhancers': 'Biostimulants',
        'Fertilizers & Soil Conditioners': 'Fertilizers',
        'Microbial & Organic Solutions': 'Microbial',
        'Pesticides & Fungicides': 'Pesticides',
        'General Agriculture & Other': 'General_Agriculture'
    })
    print("Columns after renaming and stripping:")
    print(df_fournisseurs.head()) # Print after column operations
    df_fournisseurs['Latitude'] = pd.to_numeric(df_fournisseurs['Latitude'], errors='coerce')
    df_fournisseurs['Longitude'] = pd.to_numeric(df_fournisseurs['Longitude'], errors='coerce')
    print("Columns after numeric conversion:")
    print(df_fournisseurs.head())
    df_fournisseurs = df_fournisseurs.dropna(subset=['Latitude', 'Longitude']).copy()
    print("DataFrame after dropping NaNs:")
    print(df_fournisseurs.head())
    if df_fournisseurs.empty:
        print("DataFrame is empty after processing.") # check if the dataframe is empty
except FileNotFoundError:
    print("FileNotFoundError: CSV file not found!") # Added
    df_fournisseurs = pd.DataFrame()
except Exception as e:
    print(f"An error occurred during CSV processing: {e}") # Catch other exceptions
    df_fournisseurs = pd.DataFrame()

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

def find_nearby_suppliers(user_lat, user_lon, rayon):
    if df_fournisseurs.empty:
        print("find_nearby_suppliers: DataFrame is empty at start of function.")
        return pd.DataFrame(), None

    df = df_fournisseurs.copy()
    df['Distance'] = df.apply(lambda row: calculate_distance(user_lat, user_lon, row['Latitude'], row['Longitude']), axis=1)
    nearby_suppliers = df[df['Distance'] <= rayon].sort_values('Distance')
    print("find_nearby_suppliers: Returning nearby suppliers:")
    print(nearby_suppliers)
    return nearby_suppliers, df_fournisseurs

def create_supplier_map(user_lat, user_lon, nearby_suppliers, all_suppliers):
    m = folium.Map(location=[user_lat, user_lon], zoom_start=10)
    folium.Marker(
        [user_lat, user_lon],
        popup="Your Location",
        icon=folium.Icon(color='red', icon='user')
    ).add_to(m)

    marker_cluster = MarkerCluster().add_to(m)

    if nearby_suppliers is not None and not nearby_suppliers.empty: # Check if it is not None and not empty
        for _, row in nearby_suppliers.iterrows():
            product_types = {
                'Biostimulants': row['Biostimulants'],
                'Fertilizers': row['Fertilizers'],
                'Microbial': row['Microbial'],
                'Pesticides': row['Pesticides'],
                'Other': row['Other'],
                'General_Agriculture': row['General_Agriculture']
            }
            main_type = max(product_types, key=product_types.get)
            color_map = {
                'Biostimulants': 'green',
                'Fertilizers': 'blue',
                'Microbial': 'purple',
                'Pesticides': 'orange',
                'Other': 'gray',
                'General_Agriculture': 'beige'
            }
            icon_color = color_map.get(main_type, 'gray')
            products = row['Produits'].split() if isinstance(row['Produits'], str) else []
            products_list = "<br>".join([f"• {p}" for p in products[:5]])
            if len(products) > 5:
                products_list += "<br>• ..."
            popup_content = f"""
            <div style="width:300px; font-size:12px;">
                <b>{row['Societe']}</b><br>
                <i>{row['Adresse']}</i><br><br>
                <b>Distance:</b> {row['Distance']:.1f} km<br>
                <b>Main Type:</b> {main_type}<br>
                <b>Products:</b><br>
                {products_list}
            </div>
            """
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=icon_color, icon='industry')
            ).add_to(marker_cluster)
    else:
        print("create_supplier_map: nearby_suppliers is either None or empty.")

    return m