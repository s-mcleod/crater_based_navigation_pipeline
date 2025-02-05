import pandas as pd
import math

# Function to convert latitude and longitude to selenographic coordinates
def lat_lng_to_selenographic_coordinates(lat_rad, lng_rad, polar_radius):
    X = polar_radius * math.cos(lat_rad) * math.cos(lng_rad)
    Y = polar_radius * math.cos(lat_rad) * math.sin(lng_rad)
    Z = polar_radius * math.sin(lat_rad)
    return X, Y, Z

# Read the original CSV data
df = pd.read_csv('lunar_crater_database_robbins_2018.csv')

# Constants
polar_radius = 1737.4 * 1000  # Polar radius of the Moon in meters

# Prepare the new list of data for the new CSV
new_data = []

for _, row in df.iterrows():
    crater_id = row['CRATER_ID']
    
    # Use ellipsoidal values if available; otherwise, use circular values
    lat = row['LAT_ELLI_IMG'] if not pd.isna(row['LAT_ELLI_IMG']) else row['LAT_CIRC_IMG']
    lon = row['LON_ELLI_IMG'] if not pd.isna(row['LON_ELLI_IMG']) else row['LON_CIRC_IMG']
    a_dia_km = row['DIAM_ELLI_MAJOR_IMG'] if not pd.isna(row['DIAM_ELLI_MAJOR_IMG']) else row['DIAM_CIRC_IMG']
    b_dia_km = row['DIAM_ELLI_MINOR_IMG'] if not pd.isna(row['DIAM_ELLI_MINOR_IMG']) else row['DIAM_CIRC_IMG']
    angle_deg = row['DIAM_ELLI_ANGLE_IMG'] if not pd.isna(row['DIAM_ELLI_ANGLE_IMG']) else 0
    
    # Convert latitude and longitude from degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # Convert angle from degrees to radians
    angle_rad = math.radians(angle_deg)
    
    # Calculate the selenographic coordinates
    X, Y, Z = lat_lng_to_selenographic_coordinates(lat_rad, lon_rad, polar_radius)
    
    # Append the formatted data to the new_data list
    new_data.append([crater_id, X, Y, Z, a_dia_km, b_dia_km, angle_rad])

# Create a new DataFrame for the output CSV
columns = ['ID', 'X', 'Y', 'Z', 'a_dia_km', 'b_dia_km', 'angle(rad)']
new_df = pd.DataFrame(new_data, columns=columns)

# Write the new DataFrame to a CSV file
new_df.to_csv('selenographic_lunar_crater_database_robbins.txt', index=False)

print("New CSV file 'selenographic_lunar_crater_database_robbins.txt' has been created.")