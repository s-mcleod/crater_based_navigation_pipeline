#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate cidenv

python cid_analysis.py