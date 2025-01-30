#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate cdaenv

python prediction_model.py ../data/ LDEM_-90_-45E_0_45N training_checkpoints/mask_rcnn_6.blob

conda deactivate