#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate cidenv


dir=../data/LDEM_-90_-45E_0_45N/
echo $dir


# met=PnC
# echo $met
# python main.py $dir matched_selenographic_crater_coordinates/ crater_matches/ ground_truth_flight.fli ground_truth_images/ calibration.txt --pose_method $met --scale --m_estimator

met=PnC
echo $met
python main.py $dir matched_selenographic_crater_coordinates/ crater_matches/ ground_truth_flight.fli ground_truth_images/ calibration.txt --pose_method $met --scale --propagated_position cid_estimated_position/ --m_estimator

# met=PnC
# echo $met
# python main.py $dir matched_selenographic_crater_coordinates/ crater_matches/ ground_truth_flight.fli ground_truth_images/ calibration.txt --pose_method $met --scale --propagated_position cid_estimated_position/

conda deactivate 