import os
import pandas as pd

# Define the directory path
directory = "../data/LDEM_-90_-45E_0_45N/crater_detections/"

# Process each .txt file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        
        # Read the file into a DataFrame
        data = pd.read_csv(file_path, sep=",", header=None, 
                           names=["x_centre", "y_centre", "semi_major_axis", "semi_minor_axis", 
                                  "rotation", "id", "confidence"])

        # Swap semi_major_axis and semi_minor_axis
        data[["semi_major_axis", "semi_minor_axis"]] = data[["semi_minor_axis", "semi_major_axis"]]

        # Subtract 90 from the rotation column
        data["rotation"] = pd.to_numeric(data["rotation"], errors="coerce")
        data["rotation"] -= 90

        # Write the updated data back to the file without header
        data.to_csv(file_path, sep=",", index=False, header=False)

        # Reopen the file to modify the header
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify the header (the first line)
        lines[0] = "x_centre, y_centre, semi_major_axis, semi_minor_axis, rotation, id, confidence\n"

        # Write back the modified lines (with updated header)
        with open(file_path, 'w') as file:
            file.writelines(lines)



print("Processing complete. The files have been updated.")


