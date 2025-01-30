
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from PIL import *
import cv2
import sys

import time
import math
import gc
import os

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import box_iou
import pytorch_lightning as pl
from tqdm.cli import tqdm as tq

from statistics import mean
from typing import Tuple, Dict, Iterable

from pangu_dataset_mask import load_bounding_boxes, load_projected_ellipses, compute_mask, plot_ellipses, plot_masks
from pangu_dataset_mask import CraterPredictionDataset, get_ellipse_from_mask


from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# def compute_iou(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2).sum()  # Area of overlap
#     union = np.logical_or(mask1, mask2).sum()  # Area of union
#     return intersection / union if union != 0 else 0  # IoU = intersection / union

# def get_id_maximum_iou(target, processed_mask):
#     gt_masks = target["masks"].numpy()
#     gt_ids = target["crater_id"]

#     # Find the mask with the highest IoU
#     max_iou = -1
#     best_match_id = None
#     for gt_mask, gt_id in zip(gt_masks, gt_ids):
#         iou = compute_iou(processed_mask, gt_mask)
#         if iou > max_iou:
#             max_iou = iou
#             best_match_id = gt_id

#     return best_match_id

def compute_iou(mask1, mask2):
    # Directly use bitwise operations for efficiency with binary masks
    intersection = np.bitwise_and(mask1, mask2).sum()  # More efficient intersection calculation
    union = np.bitwise_or(mask1, mask2).sum()  # More efficient union calculation
    return intersection / union if union != 0 else 0  # IoU = intersection / union

def get_id_maximum_iou(target, processed_mask):
    gt_masks = target["masks"].numpy()
    gt_ids = target["crater_id"]

    # Compute IoU for all masks
    ious = [compute_iou(processed_mask, gt_mask) for gt_mask in gt_masks]

    # Find the index of the max IoU
    max_index = np.argmax(ious)
    best_match_id = gt_ids[max_index]

    return best_match_id

model = maskrcnn_resnet50_fpn(
    pretrained = True,
)

# Change number of output classes to two ( no-crater, crater )
in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor = FastRCNNPredictor( in_features, num_classes = 2 )

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = ( param_size + buffer_size ) / 1024 ** 2
print('model size: {:.3f}MB'.format(size_all_mb))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)

path = sys.argv[3]
model.load_state_dict( torch.load( path ) )
model.eval()
model.to( device )
print( 'Loaded:', path )

# Load Robins crater database:
craters = pd.read_csv( '../data/craters_with_stats_-90_-45_0_90.csv' ).dropna()
craters = craters.loc[:, ~craters.columns.str.contains('^Unnamed')]
craters['medianDifference'] = ( craters['rimMedian'] - craters['fullMedian'] ).abs()

# Build CDA database (typically takes about a minute)
img_size = (1024, 1024)
root_dir = sys.argv[1]
ldem_name = sys.argv[2]

cd = CraterPredictionDataset(craters, img_size, root_dir, ldem_name, True, True, True)

# Return the batch as an iterable, skipping 'None' samples
def collate_fn(batch: Iterable):
    return tuple( zip( *( filter( lambda x:x is not None, batch ) ) ) )


# Define your dataset
inference_dataset = torch.utils.data.Subset(cd, range(len(cd)))  # or a custom dataset as required

# Define the DataLoader for the inference dataset
inference_data_loader = torch.utils.data.DataLoader(
    inference_dataset,
    batch_size=4,
    shuffle=False,  # No shuffling during inference
    num_workers=8,
    collate_fn=collate_fn  # Ensure this matches your dataset requirements
)

# predictions = []

# Prediction directory
output_folder = "crater_detections"
output_dir = os.path.join(root_dir+ldem_name+"/", output_folder)
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist


import gc

with torch.no_grad():
    bar = tq(inference_data_loader, desc=f"Predictions")

    for batch, (images, targets_batch) in enumerate(bar, 1):
        # Move images to the device
        images = [image.to(device) for image in images]

        # Make predictions
        predictions = model(images)

        # Process each prediction individually to minimize memory usage
        for i, prediction in enumerate(predictions):
            # Move prediction to CPU
            prediction = {k: v.to(torch.device("cpu")) for k, v in prediction.items()}

            # Prepare the output file
            instance_name = targets_batch[i]["image_id"]
            output_file = instance_name.replace("ground_truth_images", output_folder).replace("png", "txt")

            with open(output_file, "w") as f:
                # Write header
                f.write("ellipse: x_centre, y_centre, semi_major_axis, semi_minor_axis, rotation, id, confidence\n")

                # Process masks and extract ellipses
                for mask, score in zip(prediction["masks"].numpy(), prediction["scores"]):
                    im = mask[0]
                    thresh = 0.9
                    im[im >= thresh] = 1
                    im[im < thresh] = 0
                    processed_mask = (im * 255).astype(np.uint8)

                    # Extract contours from the processed mask
                    contours, _ = cv2.findContours(
                        processed_mask,
                        cv2.RETR_EXTERNAL,  # Retrieve only external contours
                        cv2.CHAIN_APPROX_SIMPLE,  # Simple contour approximation
                    )

                    # Fit ellipse and write to file
                    for contour in contours:
                        if len(contour) >= 5:
                            ellipse = cv2.fitEllipse(contour)
                            (x, y), (d1, d2), theta = ellipse
                            if d1 >= d2:
                                a = d1/2
                                b = d2/2
                            else:
                                a = d2/2
                                b = d1/2 
                                theta = (theta - 90)
                            confidence = score.item()
                            estimated_id = get_id_maximum_iou(targets_batch[i], im.astype(np.uint8))

                            # Write ellipse parameters to file
                            f.write(f"{x:.6f}, {y:.6f}, {a:.6f}, {b:.6f}, {theta:.6f}, {estimated_id}, {confidence:.6f}\n")
                            break  # Process only one contour per mask

            # Clean up memory for the current prediction
            del prediction

        # Clean up memory for the current batch
        del predictions, images
        gc.collect()


# with torch.no_grad():
#     bar = tq( inference_data_loader, desc = f"Predictions" )
    
#     for batch, ( images, targets_batch ) in enumerate( bar, 1 ):
#         images = list( image.to( device ) for image in images )

#         # Make predictions
#         p = model( images )
        
#         # Move predictions to RAM
#         p = [ { k: v.to( torch.device( 'cpu' ) ) for k, v in d.items() } for d in p ]

#         # Postprocess mask into ellipse parameters
#         for i, prediction in enumerate(p):
#             prediction['ellipse_sparse'] = []

#             # Extract filename for current instance
#             instance_name = targets_batch[i]["image_id"]
            
#             output_file = targets_batch[i]["image_id"].replace( 'ground_truth_images', output_folder ).replace( 'png', 'txt' )

#             with open(output_file, "w") as f:
#                 # Write header
#                 f.write("ellipse: x_centre, y_centre, semi_major_axis, semi_minor_axis, rotation, id, confidence\n")

#                 for j, (crater_mask, score) in enumerate(zip(prediction["masks"].numpy(), prediction["scores"])):
#                     im = crater_mask[0]
#                     thresh = 0.9
#                     im[im >= thresh] = 1
#                     im[im < thresh] = 0
#                     processed_mask = (im * 255).astype(np.uint8)

#                     # Extract contours from the processed mask
#                     contours, _ = cv2.findContours(
#                         processed_mask,
#                         cv2.RETR_EXTERNAL,  # Retrieve only external contours
#                         cv2.CHAIN_APPROX_SIMPLE,  # Simple contour approximation
#                     )
#                     # cv2.CHAIN_APPROX_NONE,  # Simple contour approximation
                    
#                     # Overlay contour points on the image
#                     for contour in contours:
#                         if len(contour) >= 5:
#                             ellipse = cv2.fitEllipse(contour)
#                             (x,y),(a,b),theta = ellipse
#                             confidence = score.item()
#                             prediction['ellipse_sparse'].append( [ x, y, a/2, b/2, theta ] )

#                             estimated_id = get_id_maximum_iou(targets_batch[i], im.astype(np.uint8))
#                             # f.write(f"{x:.6f}, {y:.6f}, {a/2:.6f}, {b/2:.6f}, {theta:.6f},{confidence:.6f}\n")
#                             f.write(f"{x:.6f}, {y:.6f}, {a/2:.6f}, {b/2:.6f}, {theta:.6f}, {estimated_id}, {confidence:.6f}\n")
#                             break
#             del prediction['masks']
#         del p, images
        
#         gc.collect()

