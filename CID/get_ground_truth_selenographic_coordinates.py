import sys
import os
import pandas as pd
import numpy as np

def process_files(input_dir, catalog_file, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the catalog into a pandas DataFrame
    catalog_columns = [
        "id", "X", "Y", "Z", "a_dia_metres", "b_dia_metres", "angle(rad)"
    ]
    # catalog = pd.read_csv(catalog_file, names=catalog_columns)
    catalog = pd.read_csv(catalog_file, names=catalog_columns, skiprows=1, dtype={"id": str})
    catalog["id"] = catalog["id"].str.strip()
    catalog = catalog.drop_duplicates(subset="id", keep="first")

    # Iterate over numbered files in input directory
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith(".txt") and file_name[:-4].isdigit():
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # Read the input file
            input_columns = [
                "x_centre", "y_centre", "semi_major_axis", 
                "semi_minor_axis", "rotation", "id"
            ]
            data = pd.read_csv(input_path, names=input_columns, skiprows=1, dtype={"id": str})
            # data = pd.read_csv(input_path, names=input_columns)
            data["id"] = data["id"].str.strip()

            # Merge the data with the catalog on the 'id' column
            merged = data.merge(catalog, left_on="id", right_on="id", how="inner")

            merged["a_dia_km"] = (merged["a_dia_km"])*1000/2
            merged["b_dia_km"] = (merged["b_dia_km"])*1000/2
   

            # Create the output structure
            output_data = merged[[
                "X", "Y", "Z", "a_dia_km", "b_dia_km", "angle(rad)", "id"
            ]]

            # Rename columns to match the required output format
            output_data.columns = [
                "X", "Y", "Z", "a_dia_km", "b_dia_km", "theta", "id"
            ]

            # Save to the output file
            output_data.to_csv(output_path, index=False, header=True)

# Example usage
input_directory = "../data/LDEM_-90_-45E_0_45N/crater_matches/"
catalogue_file = "/home/sofia/Documents/crater_based_navigation_pipeline/data/robbins_navigation_dataset.txt"
output_directory = "../data/LDEM_-90_-45E_0_45N/matched_selenographic_crater_coordinates/"

# input_directory = sys.args[1]
# catalogue_file = sys.args[2]
# output_directory = sys.args[3]

process_files(input_directory, catalogue_file, output_directory)
