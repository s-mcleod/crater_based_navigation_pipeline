# Copy CDA
cda_dir="sofia@10.18.224.95:~/Documents/crater_based_navigation_pipeline/CDA/"
files=("${cda_dir}train_mask_rcnn_sofia.ipynb" "${cda_dir}test-model-mask-rcnn.ipynb" "${cda_dir}ellipse_rcnn" "${cda_dir}pangu_dataset_mask.py" "${cda_dir}prediction_model.py" "${cda_dir}prediction_model_visualisation.ipynb" "${cda_dir}prediction_validation.ipynb" "${cda_dir}runme_predictions.sh" "${cda_dir}test-model-mask-rcnn.ipynb")
rsync -av "${files[@]}" CDA/


# Copy CID
cid_dir="sofia@10.18.224.95:~/Documents/crater_based_navigation_pipeline/CID/"
files=("${cid_dir}cid_analysis.py" "${cid_dir}cid_pecan.py" "${cid_dir}get_ground_truth_selenographic_coordinates.py" "${cid_dir}runme_cid_analysis.sh" "${cid_dir}runme.sh" "${cid_dir}src" "${cid_dir}temp_process_data.py")
rsync -av "${files[@]}" CID/

# Copy CBPE
cbpe_dir="sofia@10.18.224.95:~/Documents/crater_based_navigation_pipeline/CBPE/"
files=("${cbpe_dir}main.py" "${cbpe_dir}runme.sh" "${cbpe_dir}src")
rsync -av "${files[@]}" CBPE/

rsync -av sofia@10.18.224.95:~/Documents/crater_based_navigation_pipeline/data/convert_catalogue_to_selenographic_coordinates.py data/

git add .

git commit -m "Updating cbn pipeline on github"

git push






