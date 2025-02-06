#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate cidenv

# Get crater matches.
dir="/home/sofia/Documents/crater_based_navigation_pipeline/data/"
# crater_catalogue="robbins_navigation_dataset.txt"
crater_catalogue="selenographic_lunar_crater_database_robbins.txt"
dem="LDEM_-90_-45E_0_45N/"
crater_detections_dir="crater_detections/"
write_dir="crater_matches/"
gt_crater_detections_dir="ground_truth_projected_ellipses/"
calibration_file="calibration.txt"
flight_file="ground_truth_flight.fli"
lower_matched_percentage="0.2"
upper_matched_percentage="0.3"
write_position_dir="cid_estimated_position/"
attitude_noise_deg="0.01"

RARR_positions="simulated_RARR_positions.txt"
pos_err="6700"
python generate_RARR_position_estimates.py $dir$dem$flight_file $dir$dem$RARR_positions $pos_err

python3 cid_pecan.py --data_dir $dir$dem --catalogue_dir $dir --detections_dir $dir$dem$crater_detections_dir --write_dir $dir$dem$write_dir --gt_data_dir $dir$dem$gt_crater_detections_dir --crater_catalogue_file $dir$crater_catalogue --calibration_file $dir$dem$calibration_file --flight_file $dir$dem$flight_file --sim_RARR_pos $dir$dem$RARR_positions --lower_matched_percentage $lower_matched_percentage --upper_matched_percentage $upper_matched_percentage --write_position_dir $dir$dem$write_position_dir --attitude_noise_deg $attitude_noise_deg



# # Get the corresponding ground truth world crater locations.
# input_directory="../data/LDEM_-90_-45E_0_45N/crater_matches/"
# catalogue_file="/home/sofia/Documents/crater_based_navigation_pipeline/data/robbins_navigation_dataset.txt"
# output_directory="../data/LDEM_-90_-45E_0_45N/matched_selenographic_crater_coordinates/"

# python get_ground_truth_selenographic_coordinates.py $input_directory $catalogue_file $output_directory

# conda deactivate 
