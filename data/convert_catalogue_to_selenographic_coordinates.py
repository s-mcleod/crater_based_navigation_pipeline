import pandas as pd
import math

# Function to convert latitude and longitude to selenographic coordinates
def lat_lng_to_selenographic_coordinates(lat_rad, lng_rad, polar_radius):
    # Using the formula for converting latitude/longitude to selenographic (cartesian) coordinates.
    # Assuming selenographic coordinates (X, Y, Z) with polar radius as 1737.4 km.
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
    lat_rad = math.radians(row['LAT_ELLI_IMG'])  # Convert latitude from degrees to radians
    lon_rad = math.radians(row['LON_ELLI_IMG'])  # Convert longitude from degrees to radians
    a_dia_km = row['DIAM_ELLI_MAJOR_IMG']  # Convert diameter from meters to kilometers
    b_dia_km = row['DIAM_ELLI_MINOR_IMG']  # Convert diameter from meters to kilometers
    angle_deg = row['DIAM_ELLI_ANGLE_IMG']*math.pi/180  # The angle is already in degrees

    # Calculate the selenographic coordinates
    X, Y, Z = lat_lng_to_selenographic_coordinates(lat_rad, lon_rad, polar_radius)

    # Append the formatted data to the new_data list
    new_data.append([crater_id, X, Y, Z, a_dia_km, b_dia_km, angle_deg])

# Create a new DataFrame for the output CSV
columns = ['ID', 'X', 'Y', 'Z', 'a_dia_km', 'b_dia_km', 'angle(rad)']
new_df = pd.DataFrame(new_data, columns=columns)

# Write the new DataFrame to a CSV file
new_df.to_csv('selenographic_lunar_crater_database_robbins.csv', index=False)

print("New CSV file 'selenographic_lunar_crater_database_robbins.csv' has been created.")
