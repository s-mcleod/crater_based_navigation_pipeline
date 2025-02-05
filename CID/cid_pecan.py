import copy
import csv
import numpy as np

from src.ck_bnb import * #IMPORTANT: KEEP THIS ABOVE THE OTHER SOURCE FILES AS SOME FUNCTIONS WILL BE OVERWRITTEN.
from src.get_data import *
from src.utils import *


from scipy.linalg import eig

# import matplotlib
# matplotlib.use('TkAgg')

import warnings
from numba.core.errors import NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

from numba import njit, prange, cuda
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import csv
import time
from joblib import Parallel, delayed
import re


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os






def plot_ellipses_with_ids(imaged_params, craters_indices, imaged_params_gt, craters_indices_gt):
    """
    Plots and saves ellipses for detected and ground-truth craters, assigning the same color for the same ID.
    Only ground-truth craters with IDs matching detected craters are plotted.
    
    Parameters:
        imaged_params (list): List of detected crater parameters [x, y, major_axis, minor_axis, angle].
        craters_indices (list): List of detected crater IDs.
        imaged_params_gt (list): List of ground-truth crater parameters [x, y, major_axis, minor_axis, angle].
        craters_indices_gt (list): List of ground-truth crater IDs.
    """
    # Create a unique color for each crater ID
    unique_ids = np.unique(craters_indices)
    id_to_color = {crater_id: np.random.rand(3,) for crater_id in unique_ids}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1024)  # Adjust according to your data range
    ax.set_ylim(0, 1024)  # Adjust according to your data range

    # Plot detected craters
    for params, crater_id in zip(imaged_params, craters_indices):
        x, y, major_axis, minor_axis, angle = params
        color = id_to_color.get(crater_id, (0, 0, 0))  # Default to black if no color is found
        ellipse = Ellipse((x, y), width=major_axis, height=minor_axis, angle=np.degrees(angle),
                          edgecolor=color, facecolor='none', linewidth=2)
        ax.add_patch(ellipse)

    # Plot ground-truth craters (only if their IDs exist in the detected IDs)
    for params, crater_id in zip(imaged_params_gt, craters_indices_gt):
        if crater_id in id_to_color:  # Check if the ground-truth ID exists in detected IDs
            x, y, major_axis, minor_axis, angle = params
            color = id_to_color[crater_id]
            ellipse = Ellipse((x, y), width=major_axis, height=minor_axis, angle=np.degrees(angle),
                              edgecolor=color, linestyle='dashed', facecolor='none', linewidth=2)
            ax.add_patch(ellipse)

    # Set axis properties
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Detected and Ground-Truth Craters')
    plt.grid(True)

    # Display the plot
    plt.show()

    # Save the plot to a file
    output_path = "img.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")




def get_files_in_dir(dir, ext):
    if dir[-1] != "/":
        dir += "/"
    files = glob.glob(dir+"*."+ext)
    files = [file[len(dir):] for file in files]
    return files

def find_min_and_indices(min_obj_val, opt_pos_mat, opt_att_mat):
    """
    Find the minimum value and its indices from a given 2D matrix.
    Also, retrieve the corresponding opt_pos and opt_att matrices for those indices.

    Parameters:
    - min_obj_val: 2D matrix from which the minimum value and its indices are to be found
    - opt_pos_mat: Matrix corresponding to opt_pos
    - opt_att_mat: Matrix corresponding to opt_att

    Returns:
    - min_val: Minimum value from min_obj_val
    - min_indices: Indices of the minimum value in min_obj_val
    - opt_pos: Corresponding opt_pos matrix value
    - opt_att: Corresponding opt_att matrix value
    """
    min_val = np.min(min_obj_val)
    min_indices = np.unravel_index(np.argmin(min_obj_val), min_obj_val.shape)
    opt_pos = opt_pos_mat[min_indices]
    opt_att = opt_att_mat[min_indices]

    return min_val, min_indices, opt_pos, opt_att


def find_correspondences(CW_params, db_CW_params):
    # Initialize an empty list to store the indices of the correspondences
    correspondence_indices = []

    # Loop through each row in CW_params
    for i in range(CW_params.shape[0]):
        # Calculate the distance between the current row in CW_params and each row in db_CW_params
        distances = np.linalg.norm(db_CW_params - CW_params[i, :], axis=1)

        # Find the index of the minimum distance
        min_index = np.argmin(distances)

        # Append the index to the list of correspondences
        correspondence_indices.append(min_index)

    return np.array(correspondence_indices)




@njit
def compute_ellipse_distance_matrix(db_CW_conic_inv, db_CW_Hmi_k, P_mc, CC_params, neighbouring_craters_id):
    # Initialize the distance matrix
    pdist_mat = np.ones((CC_params.shape[0], len(neighbouring_craters_id))) * np.inf

    for ncidx, ncid in enumerate(neighbouring_craters_id):
        curr_db_CW_conic_inv = db_CW_conic_inv[ncid]

        # Project it down
        legit_flag, curr_A = conic_from_crater_cpu(curr_db_CW_conic_inv, db_CW_Hmi_k[ncid], P_mc)

        if not(legit_flag):
            continue

        # Extract xy first
        curr_A_params = extract_ellipse_parameters_from_conic(curr_A)

        if np.any(np.isnan(np.array(curr_A_params[1:]))):
            continue

        if not(curr_A_params[0]):
            continue

        for cc_id in range(CC_params.shape[0]):
            scaled_curr_A_params = np.array(curr_A_params[1:])
            x1,y1,a1,b1,t1 = scaled_curr_A_params
            x2,y2,a2,b2,t2 = CC_params[cc_id]
            # pdist_mat[cc_id, ncidx] = np.linalg.norm(scaled_curr_A_params - CC_params[cc_id]) #TODO
            pos_scale = 1
            radii_scale = 1.5
            theta_scale = 1
            pdist_mat[cc_id, ncidx] = pos_scale*np.linalg.norm(np.array([x1,y1]) - np.array([x2,y2])) + radii_scale*np.linalg.norm(np.array([a1,b1]) - np.array([a2,b2])) + theta_scale*np.linalg.norm(np.array([t1]) - np.array([t2]))

    return pdist_mat


def p1e_solver(CW_params, CW_ENU, Rw_c, CC_params, K, craters_id):
    # curr_CW_param = CW_params[craters_id]
    curr_CW_param = CW_params

    eps = 1
    Aell = np.diag([1 / curr_CW_param[3] ** 2, 1 / curr_CW_param[4] ** 2, 1 / eps ** 2])

    # Convert Aell to Aworld
    # Rw_ell = CW_ENU[craters_id]
    Rw_ell = CW_ENU

    # gt_att is the extrinsic, aka, it converts world's coord to cam's coord.
    Re_cam = Rw_c.T @ Rw_ell
    Acam = Re_cam @ Aell @ Re_cam.T

    # Backproject conic
    curr_conic = CC_params[craters_id]
    R_ellipse = np.zeros([2, 2])
    R_ellipse[0, 0] = np.cos(curr_conic[4])
    R_ellipse[0, 1] = -np.sin(curr_conic[4])
    R_ellipse[1, 0] = np.sin(curr_conic[4])
    R_ellipse[1, 1] = np.cos(curr_conic[4])
    R_ellipse = -R_ellipse
    Kc = np.linalg.inv(K) @ np.transpose(np.array([curr_conic[0], curr_conic[1], 1])) * K[0, 0]
    Ec = np.transpose(np.array([0, 0, 0]))

    # Compute Uc, Vc, Nc
    Uc = np.append(R_ellipse[:, 0], 0)
    Vc = np.append(R_ellipse[:, 1], 0)
    Nc = np.array([0, 0, 1])

    # Compute M
    M = np.outer(Uc, Uc) / curr_conic[2] ** 2 + np.outer(Vc, Vc) / curr_conic[3] ** 2

    # Compute W
    W = Nc / np.dot(Nc, (Kc - Ec))

    # Compute P
    P = np.identity(3) - np.outer((Kc - Ec), W)

    # Compute Q
    Q = np.outer(W, W)

    # Compute Bcam
    Bcam = P.T @ M @ P - Q

    if np.isnan(Bcam).any():
        print(curr_conic, CC_params[craters_id])

    V = eig(Acam, Bcam, left=True, right=False)[1]

    D = np.real(eig(Acam, Bcam)[0])

    sameValue, uniqueValue, uniqueIdx = differentiate_values(D)

    sigma_1 = uniqueValue
    sigma_2 = sameValue

    d1 = V[:, uniqueIdx].T
    d1 = d1 / np.linalg.norm(d1)

    with np.errstate(invalid='ignore'):
        k = np.sqrt(np.trace(np.linalg.inv(Acam)) - (1 / sigma_2) * np.trace(np.linalg.inv(Bcam)))

    delta_cam_est = k * d1
    delta_cam_est_flip = -k * d1

    delta_cam_world_est = Rw_c @ delta_cam_est.T
    E_w_est = curr_CW_param[0:3] + delta_cam_world_est

    delta_cam_world_flip_est = Rw_c @ delta_cam_est_flip.T
    E_w_flip_est = curr_CW_param[0:3] + delta_cam_world_flip_est

    return E_w_est, E_w_flip_est


def db_id_mask_with_errors(CW_params, K, pos_est, att_est, pos_err, att_err_deg, num_samples=10):
    img_w, img_h = K[0, 2] * 2, K[1, 2] * 2  # Assuming the principal point is at (w/2, h/2)
    neighbouring_craters_id = np.arange(CW_params.shape[0])
    combined_mask = np.zeros(len(neighbouring_craters_id), dtype=bool)
    
    for _ in range(num_samples):
        # Sample position error within bounds
        pos_perturbation = pos_err * (2 * np.random.rand(3) - 1)  # Uniform perturbation
        perturbed_pos = pos_est + pos_perturbation
        
        # Sample attitude error within bounds
        att_perturbation = R.from_euler('xyz', att_err_deg * (2 * np.random.rand(3) - 1), degrees=True).as_matrix()
        perturbed_att = att_perturbation @ att_est
        
        # Compute projection matrix
        extrinsic = np.zeros((3, 4))
        extrinsic[0:3, 0:3] = perturbed_att
        extrinsic[:, 3] = perturbed_att @ -perturbed_pos
        P_mc = K @ extrinsic
        
        # Project 3D points onto the image plane
        projected_3D_points = P_mc @ np.hstack([CW_params[:, 0:3], np.ones((len(neighbouring_craters_id), 1))]).T
        points_on_img_plane = np.array([
            projected_3D_points[0, :] / projected_3D_points[2, :],
            projected_3D_points[1, :] / projected_3D_points[2, :]
        ])
        
        within_img_valid_indices = np.where((points_on_img_plane[0, :] >= 0) &
                                            (points_on_img_plane[0, :] <= img_w) &
                                            (points_on_img_plane[1, :] >= 0) &
                                            (points_on_img_plane[1, :] <= img_h) &
                                            ~np.isnan(points_on_img_plane[0, :]) &
                                            ~np.isnan(points_on_img_plane[1, :]))[0]
        
        fil_ncid = neighbouring_craters_id[within_img_valid_indices]
        cam_pos = -extrinsic[0:3, 0:3].T @ extrinsic[0:3, 3]
        
        # Check if craters are visible on the sphere
        _, fil_ncid = visible_points_on_sphere(CW_params[:, 0:3], np.array([0, 0, 0]),
                                               np.linalg.norm(CW_params[0, 0:3]),
                                               cam_pos, fil_ncid)
        
        # Update mask
        combined_mask |= np.isin(neighbouring_craters_id, fil_ncid)
    
    return combined_mask



def db_id_mask(CW_params, K, pos_est, att_est):
    extrinsic = np.zeros((3,4))
    
    extrinsic[0:3,0:3] = att_est
    extrinsic[:,3] = att_est @ -pos_est
    P_mc = K @ extrinsic

    #  Only process the rest if the pre-screen test is passed
    neighbouring_craters_id = np.arange(CW_params.shape[0])

    # 1) project all 3D points onto the image plane
    projected_3D_points = P_mc @ np.hstack(
        [CW_params[neighbouring_craters_id, 0:3], np.ones((len(neighbouring_craters_id), 1))]).T
    points_on_img_plane = np.array([projected_3D_points[0, :] / projected_3D_points[2, :],
                                    projected_3D_points[1, :] / projected_3D_points[2, :]])

    within_img_valid_indices = np.where((points_on_img_plane[0, :] >= 0) &
                                        (points_on_img_plane[0, :] <= img_w) &
                                        (points_on_img_plane[1, :] >= 0) &
                                        (points_on_img_plane[1, :] <= img_h) &
                                        ~np.isnan(points_on_img_plane[0, :]) &
                                        ~np.isnan(points_on_img_plane[1, :]))[0]

    fil_ncid = neighbouring_craters_id[within_img_valid_indices]

    cam_pos = -extrinsic[0:3, 0:3].T @ extrinsic[0:3, 3]

    # check if the crater is visible to the camera
    _, fil_ncid = visible_points_on_sphere(CW_params[:, 0:3], np.array([0, 0, 0]),
                                           np.linalg.norm(CW_params[0, 0:3]),
                                           cam_pos, fil_ncid)
    
    mask = np.isin(neighbouring_craters_id, fil_ncid)
    return mask

# def read_crater_database(craters_database_text_dir):
#     with open(craters_database_text_dir, "r") as f:
#         lines = f.readlines()[1:]  # ignore the first line
#     lines = [(re.split(r',\s*', i)) for i in lines]
#     lines = np.array(lines)

#     ID = lines[:, 0]
#     lines = np.float64(lines[:, 1:])

#     db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, db_L_prime = get_craters_world(lines)
#     crater_center_point_tree = cKDTree(db_CW_params[:, 0:3])
#     return db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, ID, crater_center_point_tree, db_L_prime
    

def read_crater_database(craters_database_text_dir, cached_filename, overwrite_catalogue=False):
    if os.path.exists(cached_filename) and not overwrite_catalogue:  # Load from cache if available
        data = np.load(cached_filename, allow_pickle=True)
        return (data['db_CW_params'], data['db_CW_conic'], data['db_CW_conic_inv'], 
                data['db_CW_ENU'], data['db_CW_Hmi_k'], data['ID'], 
                cKDTree(data['db_CW_params'][:, 0:3]), data['db_L_prime'])

    with open(craters_database_text_dir, "r") as f:
        lines = f.readlines()[1:]  # Ignore the first line
    lines = [re.split(r',\s*', i.strip()) for i in lines]
    lines = np.array(lines, dtype=object)  # Use dtype=object to preserve strings

    ID = lines[:, 0]
    lines = np.array(lines[:, 1:], dtype=np.float64)  # Convert numerical data

    db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, db_L_prime = get_craters_world(lines)
    # Save to cache
    np.savez_compressed(cached_filename, 
                        db_CW_params=db_CW_params, 
                        db_CW_conic=db_CW_conic, 
                        db_CW_conic_inv=db_CW_conic_inv, 
                        db_CW_ENU=db_CW_ENU, 
                        db_CW_Hmi_k=db_CW_Hmi_k, 
                        ID=ID, 
                        db_L_prime=db_L_prime)

    crater_center_point_tree = cKDTree(db_CW_params[:, 0:3])
    return db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, ID, crater_center_point_tree, db_L_prime
    

def strip_symbols(s, symbols):
    for symbol in symbols:
        s = s.replace(symbol, '')
    return s




def get_imaged_gt_craters(data_dir,sorted_crater_cam_filenames, degrees = False, only_ids=None):
    all_detected_craters = []
    all_crater_ids = []
    for file_name in sorted_crater_cam_filenames:
        detected_craters_file = os.path.abspath(data_dir+file_name)
        f = open(detected_craters_file,"r")
        lines = f.readlines()[1:] #ignore the first line
        lines = [re.split(r',\s*', i) for i in lines]

        if degrees:
            imaged_craters = np.array([np.array([float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4])*np.pi/180]) for i in lines])
        else:
            imaged_craters = np.array([np.array([float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4])]) for i in lines])
        crater_ids = np.array([i[5][:-1] for i in lines])
        confidence_mask = np.array(crater_ids != "None", dtype=bool)

        if only_ids:
            id_mask = np.isin(crater_ids, ID)

            # Combined mask
            combined_mask = confidence_mask & id_mask

            imaged_craters = imaged_craters[combined_mask]
            imaged_craters = [np.array(i) for i in imaged_craters]

            all_detected_craters.append(imaged_craters)
            all_crater_ids.append(crater_ids[combined_mask])

        else:
            imaged_craters = imaged_craters[confidence_mask]
            imaged_craters = [np.array(i) for i in imaged_craters]

            all_detected_craters.append(imaged_craters)
            all_crater_ids.append(crater_ids[confidence_mask])

    return all_detected_craters, all_crater_ids   

def get_imaged_craters(data_dir,sorted_crater_cam_filenames, degrees = False, confidence_threshold = 0):
    all_detected_craters = []
    all_crater_ids = []
    for file_name in sorted_crater_cam_filenames:
        detected_craters_file = os.path.abspath(data_dir+file_name)
        f = open(detected_craters_file,"r")
        lines = f.readlines()[1:] #ignore the first line
        lines = [re.split(r',\s*', i) for i in lines]

        if degrees:
            imaged_craters = np.array([np.array([float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4])*np.pi/180]) for i in lines])
        else:
            imaged_craters = np.array([np.array([float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4])]) for i in lines])
        crater_ids = np.array([i[5] for i in lines])
        crater_dection_confidence = np.array([float(i[6]) for i in lines])
    

        confidence_mask = np.array(crater_dection_confidence >= confidence_threshold, dtype=bool)

        imaged_craters = imaged_craters[confidence_mask]
        imaged_craters = [np.array(i) for i in imaged_craters]

        all_detected_craters.append(imaged_craters)
        all_crater_ids.append(crater_ids[confidence_mask])
    return all_detected_craters, all_crater_ids

def get_imaged_craters_filter_id(data_dir,sorted_crater_cam_filenames, degrees, confidence_threshold, ID=None):
    all_detected_craters = []
    all_crater_ids = []
    for file_name in sorted_crater_cam_filenames:
        detected_craters_file = os.path.abspath(data_dir+file_name)
        f = open(detected_craters_file,"r")
        lines = f.readlines()[1:] #ignore the first line
        lines = [re.split(r',\s*', i) for i in lines]
        if degrees:
            imaged_craters = np.array([np.array([float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4])*np.pi/180]) for i in lines])
        else:
            imaged_craters = np.array([np.array([float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4])]) for i in lines])
        crater_ids = np.array([i[5] for i in lines])
        crater_dection_confidence = np.array([float(i[6]) for i in lines])
    
        # Ensure the ellipse is valid
        semi_major_axis = np.array([float(i[2]) for i in lines])
        semi_minor_axis = np.array([float(i[3]) for i in lines])
        semi_major_axis_mask = np.array(semi_major_axis > 0, dtype=bool)
        semi_minor_axis_mask = np.array(semi_minor_axis > 0, dtype=bool)
        axis_mask = semi_major_axis_mask & semi_minor_axis_mask
 
        confidence_mask = np.array(crater_dection_confidence >= confidence_threshold, dtype=bool)

        id_mask = np.ones(len(crater_ids), dtype=bool)

        if ID is not None:
            ID_set = set(ID)  # Convert to a set (fast lookup)
            id_mask = np.array([crater_id in ID_set for crater_id in crater_ids])
            # id_mask = np.isin(crater_ids, ID) #TODO: we might want to change this
        combined_mask = confidence_mask & id_mask & axis_mask

        imaged_craters = imaged_craters[combined_mask]

        imaged_craters = [np.array(i) for i in imaged_craters]

        all_detected_craters.append(imaged_craters)
        all_crater_ids.append(crater_ids[combined_mask])
    return all_detected_craters, all_crater_ids

def get_extrinsic_from_flight_file(flight_file, not_pangu, attitude_noise_deg):
    f = open(flight_file, 'r')
    lines = f.readlines()
    lines = [i.split() for i in lines]
    camera_extrinsics = np.zeros([len(lines), 3, 4])
    for i, line in enumerate(lines):
        # Camera pose line is prefixed with "start" and has structure -> x, y, z, yaw, pitch, roll respectively
        if len(line) > 0 and line[0] == "start":
            pose = np.float_(line[1:])
            x, y, z, yaw, pitch, roll = pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
            
            if not_pangu:
                Tm_c = R.from_euler('ZYX',np.array([yaw, pitch, roll]),degrees=True).as_matrix()
            else:
                R_w_ci_intrinsic = R.from_euler('ZXZ',np.array([0,-90,0]),degrees=True).as_matrix()
                R_ci_cf_intrinsic = R.from_euler('ZXZ',np.array([yaw, pitch, 0]),degrees=True).as_matrix()
                R_c_intrinsic = np.dot(R_ci_cf_intrinsic, R_w_ci_intrinsic)
                R_w_c_extrinsic = np.linalg.inv(R_c_intrinsic)
                R_c_roll_extrinsic = R.from_euler('xyz',np.array([0,0,roll]),degrees=True).as_matrix()
                R_w_c = np.dot(R_c_roll_extrinsic,R_w_c_extrinsic)
                Tm_c = R_w_c

            position = ([x, y, z])
            rm = np.array(list(position)) # position of camera in the moon reference frame
            rc = np.dot(Tm_c, -1*rm) # position of camera in the camera reference frame
            so3 = np.empty([3,4])
            so3[0:3, 0:3] = Tm_c
            so3[0:3,3] = rc 
            
            camera_extrinsics[i] = so3
    return camera_extrinsics



def testing_data_reading(dir):
    with open(dir, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = list(reader)

    camera_extrinsic = np.zeros([len(data), 3, 4])
    camera_pointing_angle = np.zeros(len(data))
    heights = np.zeros(len(data))
    noise_levels = np.zeros(len(data))
    remove_percentages = np.zeros(len(data))
    add_percentages = np.zeros(len(data))
    att_noises = np.zeros(len(data))  # Att_noise is always one value
    noisy_cam_orientations = np.zeros([len(data), 3, 3])  # Noisy cam orientation is always a 3x3 matrix

    imaged_params = []
    noisy_imaged_params = []
    crater_indices = []

    for row_id, row in enumerate(data):
        # Extract Camera Extrinsic matrix
        row_0 = row[0].split('\n')
        curr_cam_ext = np.zeros([3, 4])
        for i in range(len(row_0)):
            curr_row = strip_symbols(row_0[i], ['[', ']'])
            curr_array = np.array([float(value) for value in curr_row.split()]).reshape(1, 4)
            curr_cam_ext[i] = curr_array
        camera_extrinsic[row_id] = curr_cam_ext

        # Extract Camera Pointing Angle
        camera_pointing_angle[row_id] = float(row[1])

        # Extract Imaged Conics matrices
        curr_imaged_params = [np.array(conic) for conic in eval(row[2])]
        imaged_params.append(curr_imaged_params)

        # Extract Imaged Conics matrices
        curr_imaged_params = [np.array(conic) for conic in eval(row[3])]
        noisy_imaged_params.append(curr_imaged_params)

        # Extract Crater Indices
        # crater_indices.append(literal_eval(row[4]))
        curr_conic_indices = np.array(eval(row[4]))
        crater_indices.append(curr_conic_indices)

        # Extract Height
        heights[row_id] = float(row[5])

        # Extract Noise Level
        noise_levels[row_id] = float(row[6])

        # Extract Remove Percentage
        remove_percentages[row_id] = float(row[7])

        # Extract Add Percentage
        add_percentages[row_id] = float(row[8])

        # Extract Attitude Noise
        att_noises[row_id] = float(row[9])

        # Extract Noisy Camera Orientation
        # noisy_cam_orientations[row_id] = np.array(literal_eval(row[10]))
        row_10 = row[10].split('\n')
        curr_nc = np.zeros([3, 3])
        for i in range(len(row_10)):
            curr_row = strip_symbols(row_10[i], ['[', ']'])
            curr_array = np.array([float(value) for value in curr_row.split()]).reshape(1, 3)
            curr_nc[i] = curr_array
        noisy_cam_orientations[row_id] = curr_nc

    return camera_extrinsic, camera_pointing_angle, imaged_params, noisy_imaged_params, crater_indices, \
           heights, noise_levels, remove_percentages, add_percentages, att_noises, noisy_cam_orientations



def log_result(matched_ids, gt_ids, result_dir, i):

    # Initialize counts
    TP = FP = FN = TN = 0

    # Compute TP, FP, FN, and TN
    for m_id, gt_id in zip(matched_ids, gt_ids):
        if m_id != 'None' and gt_id != 'None':
            if m_id == gt_id:
                TP += 1
            else:
                FP += 1
        elif m_id == 'None' and gt_id != 'None':
            FN += 1
        elif m_id != 'None' and gt_id == 'None':
            FP += 1
        elif m_id == 'None' and gt_id == 'None':
            TN += 1

    # Compute rates
    FMR = FP / len([gt_id for gt_id in gt_ids])
    FNR = FN / len([gt_id for gt_id in gt_ids if gt_id != 'None'])
    
    matching_rate = TP / len([gt_id for gt_id in gt_ids if gt_id != 'None'])


    # Format the results in a single line
    result_str = ("Testing ID: {} | Matched IDs: {} | Matching Rate: {:.2f} | False Matching Rate: {:.2f} | False Negative Rate: {:.2f} \n").format(
        i, ', '.join(str(id) for id in matched_ids), matching_rate, FMR, FNR)
    
    print(result_str)

    # Open the file in append mode and write the result
    with open(result_dir, 'a') as file:
        file.write(result_str)

    return matched_ids



@njit
def find_nearest_neighbors(dist_matrix):
    M, N = dist_matrix.shape

    # Placeholder for nearest neighbors for each row
    nearest_neighbors_id = -np.ones(M, dtype=np.int32)
    nearest_neighbors_val = np.ones(M, dtype=np.float32) * np.inf

    # Flatten and argsort manually
    flat_size = M * N
    flat_distances = np.empty(flat_size, dtype=dist_matrix.dtype)
    for i in range(M):
        for j in range(N):
            flat_distances[i * N + j] = dist_matrix[i, j]

    sorted_indices = np.argsort(flat_distances)
    assigned_columns = set()
    for k in range(flat_size):
        index = sorted_indices[k]
        i = index // N
        j = index % N

        if nearest_neighbors_id[i] == -1 and j not in assigned_columns:
            nearest_neighbors_id[i] = j
            nearest_neighbors_val[i] = dist_matrix[i, j]
            assigned_columns.add(j)

        # # If there are more rows than columns, a column can be chosen multiple times.
        # # Otherwise, a column should be chosen at most once.
        # if nearest_neighbors_id[i] == -1 and (M > N or not np.any(nearest_neighbors_id == j)):
        #     nearest_neighbors_id[i] = j
        #     nearest_neighbors_val[i] = dist_matrix[i, j]

        # Break when all rows have been assigned
        if not np.any(nearest_neighbors_id == -1):
            break

    return nearest_neighbors_id, nearest_neighbors_val

def display_projected_ellipses(CC_params, CW_ids, CW_conic_inv, CW_Hmi_k, K, opt_matched_ids, gt_att, opt_cam_pos):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1024)  # Adjust according to your data range
    ax.set_ylim(0, 1024)  # Adjust according to your data range

    for i, idn in enumerate(opt_matched_ids):
        if idn != "None":
            print("id:",idn)
            index = CW_ids.tolist().index(idn)
            so3 = np.zeros([3, 4])
            so3[:,0:3] = gt_att
            so3[:,3] = gt_att @ -opt_cam_pos
            P_mc = K @ so3

            _, curr_A = conic_from_crater_cpu(CW_conic_inv[index], CW_Hmi_k[index], P_mc)
            _, x1, y1, a1, b1, theta_rad1 = extract_ellipse_parameters_from_conic(curr_A)
            print("pred:",x1, y1, a1, b1, theta_rad1)
            # Draw predicted ellipse
            ellipse_pred = Ellipse((x1, y1), width=2*a1, height=2*b1, angle=np.degrees(theta_rad1),
                                   edgecolor='blue', facecolor='none', linewidth=2, label='Predicted Ellipse' if i == 0 else "")
            ax.add_patch(ellipse_pred)

            # Draw ground truth ellipse
            x2, y2, a2, b2, theta_rad2 = CC_params[i]
            print("dete:",x2, y2, a2, b2, theta_rad2)
            ellipse_gt = Ellipse((x2, y2), width=2*a2, height=2*b2, angle=np.degrees(theta_rad2),
                                 edgecolor='green', facecolor='none', linewidth=2, linestyle='--', label='Ground Truth Ellipse' if i == 0 else "")
            ax.add_patch(ellipse_gt)

            print(np.linalg.norm(np.array([x1, y1, a1, b1, theta_rad1])-np.array([x2, y2, a2, b2, theta_rad2])))
            print(np.linalg.norm(np.array([x1, y1])-np.array([x2, y2])))
            print(np.linalg.norm(np.array([a1, b1])-np.array([a2, b2])))
            print(np.linalg.norm(np.array([theta_rad1])-np.array([theta_rad2])))
            print()
    
    # Set axis properties
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Detected and Ground-Truth Craters')
    plt.grid(True)

    # Display the plot
    plt.show()

    # Save the plot to a file
    output_path = "pred_img.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")


def main_func(db_cw_id, CW_params, CW_ENU, CW_L_prime, Rw_c, CC_params, K, cc_id, gt_att,
              CW_conic_inv, CW_Hmi_k,
              px_thres, ab_thres, deg_thres,
              eld_thres, img_w, img_h):

    opt_num_matches = 0
    opt_cam_pos = np.array([0, 0, 0])
    opt_matched_ids = np.zeros(CC_params.shape[0])

    E_w_est, E_w_flip_est = p1e_solver(CW_params[db_cw_id], CW_L_prime[db_cw_id], Rw_c, CC_params, K,
                                       cc_id)

    rc_pos = gt_att @ -E_w_est
    rc_neg = gt_att @ -E_w_flip_est

    so3_pos = np.zeros([3, 4])
    so3_neg = np.zeros([3, 4])

    so3_pos[:, 0:3] = gt_att
    so3_neg[:, 0:3] = gt_att

    so3_pos[:, 3] = rc_pos
    so3_neg[:, 3] = rc_neg

    P_mc_pos = K @ so3_pos
    P_mc_neg = K @ so3_neg

    # chirality test here
    curr_crater_center_homo = np.array([CW_params[db_cw_id, 0:3]])
    curr_crater_center_homo = np.append(curr_crater_center_homo, 1)
    proj_pos = P_mc_pos @ curr_crater_center_homo.T
    proj_neg = P_mc_neg @ curr_crater_center_homo.T

    if proj_pos[2] > 0:  # chiraility test
        P_mc = P_mc_pos
        so3 = so3_pos
        cam_pos = E_w_est
    elif proj_neg[2] > 0:
        P_mc = P_mc_neg
        so3 = so3_neg
        cam_pos = E_w_flip_est
    else:
        return opt_num_matches, opt_matched_ids, opt_cam_pos, []
        # continue

    # compute the distance here
    legit_flag, curr_A = conic_from_crater_cpu(CW_conic_inv[db_cw_id], CW_Hmi_k[db_cw_id], P_mc)
    # Extract xy first
    if not (legit_flag):
        return opt_num_matches, opt_matched_ids, opt_cam_pos, []
        # continue

    curr_A_params = extract_ellipse_parameters_from_conic(curr_A)

    # _, x1, y1, a1, b1, t1 = curr_A_params 
    # x2, y2, a2, b2, t2 = CC_params[cc_id]
    # print("gt:",x1, y1, a1, b1, t1, t1*180/math.pi,"\n","pd:",x2, y2, a2, b2, t2, t2*180/math.pi)


    nextStageFlag = False
    # a pre-screen test here, the thresholds are set to be loose so that this step allows more pairs to be passed
    if curr_A_params[0]:
        px_dev = np.linalg.norm((curr_A_params[1:3]) - (CC_params[cc_id, 0:2]))
        a_dev = np.abs(curr_A_params[3] - CC_params[cc_id, 2]) / CC_params[cc_id, 2]
        b_dev = np.abs(curr_A_params[4] - CC_params[cc_id, 3]) / CC_params[cc_id, 3]
        phi_dev = np.abs(curr_A_params[-1] - CC_params[cc_id, -1])

        if px_dev < px_thres and a_dev < ab_thres and b_dev < ab_thres and phi_dev < np.radians(deg_thres):
            nextStageFlag = True

    if not (nextStageFlag):
        return opt_num_matches, opt_matched_ids, opt_cam_pos, []
        # continue

    # Only process the rest if the pre-screen test is passed
    neighbouring_craters_id = np.arange(CW_params.shape[0])

    # 1) project all 3D points onto the image plane
    projected_3D_points = P_mc @ np.hstack(
        [CW_params[neighbouring_craters_id, 0:3], np.ones((len(neighbouring_craters_id), 1))]).T
    points_on_img_plane = np.array([projected_3D_points[0, :] / projected_3D_points[2, :],
                                    projected_3D_points[1, :] / projected_3D_points[2, :]])

    within_img_valid_indices = np.where((points_on_img_plane[0, :] >= 0) &
                                        (points_on_img_plane[0, :] <= img_w) &
                                        (points_on_img_plane[1, :] >= 0) &
                                        (points_on_img_plane[1, :] <= img_h) &
                                        ~np.isnan(points_on_img_plane[0, :]) &
                                        ~np.isnan(points_on_img_plane[1, :]))[0]

    fil_ncid = neighbouring_craters_id[within_img_valid_indices]

    # check if the crater is visible to the camera
    _, fil_ncid = visible_points_on_sphere(CW_params[:, 0:3], np.array([0, 0, 0]),
                                           np.linalg.norm(CW_params[0, 0:3]),
                                           cam_pos, fil_ncid)

    if len(fil_ncid) == 0:
        return opt_num_matches, opt_matched_ids, opt_cam_pos, []

    try:
        el_dist_mat = compute_ellipse_distance_matrix(CW_conic_inv, CW_Hmi_k, P_mc, CC_params,
                                                      fil_ncid)
    except:
        el_dist_mat = np.ones((CC_params.shape[0], len(neighbouring_craters_id))) * np.inf

    nearest_neighbors_idx, nearest_neighbors_val = find_nearest_neighbors(el_dist_mat)
    # print(nearest_neighbors_idx)
    # print(nearest_neighbors_val)
    closest_neighbouring_ids = [fil_ncid[idx] for idx in nearest_neighbors_idx]

    # first level test, if it passes, go to second level
    matched_count = np.sum(nearest_neighbors_val <= eld_thres)
    if not (matched_count > lower_matched_percentage * CC_params.shape[0]):
        return opt_num_matches, opt_matched_ids, opt_cam_pos, nearest_neighbors_val
        # continue

    # Below is the refinement step to maximize the number of matches
    len_CC_params = CC_params.shape[0]
    CW_matched_ids = []
    CW_params_sub = np.zeros([len_CC_params, CW_params.shape[1]])
    CW_ENU_sub = np.zeros([len_CC_params, CW_ENU.shape[1], CW_ENU.shape[2]])
    CW_L_prime_sub = np.zeros([len_CC_params, CW_L_prime.shape[1], CW_L_prime.shape[2]])

    for j in range(CC_params.shape[0]):
        CW_matched_ids.append(ID[closest_neighbouring_ids[j]])
        CW_params_sub[j] = CW_params[closest_neighbouring_ids[j]]
        CW_ENU_sub[j] = CW_ENU[closest_neighbouring_ids[j]]
        CW_L_prime_sub[j] = CW_L_prime[closest_neighbouring_ids[j]]

    opt_num_matches, opt_matched_ids, opt_cam_pos, _ = refinement(CW_params, CW_conic_inv, CW_Hmi_k, ID,
                                                                    CW_params_sub, CW_L_prime_sub,
                                                                    Rw_c, CC_params, K,
                                                                    eld_thres,
                                                                    img_w, img_h)
    return opt_num_matches, opt_matched_ids, opt_cam_pos, nearest_neighbors_val


def refinement(CW_params, CW_conic_inv, CW_Hmi_k, ID,
                    CW_params_sub, CW_L_prime_sub,
                    Rw_c, CC_params, K, eld_thres,
                    img_w, img_h):
    '''
    #### here we look for the pair that leads to the highest consensus matches
    :param CW_params_sub: CW correspondence for CC_params
    :param Rw_c:
    :param CC_params:
    :param K:
    :return:
    '''
    gt_att = Rw_c.T
    so3_pos = np.zeros([3, 4])
    so3_neg = np.zeros([3, 4])

    so3_pos[:, 0:3] = gt_att
    so3_neg[:, 0:3] = gt_att
    max_match_count = 0
    opt_matched_ids = [[] for _ in range(CC_params.shape[0])]
    opt_cam_pos = np.array([0, 0, 0])
    for cc_id in range(CC_params.shape[0]):
        E_w_est, E_w_flip_est = p1e_solver(CW_params_sub[cc_id], CW_L_prime_sub[cc_id], Rw_c, CC_params, K,
                                           cc_id)

        rc_pos = gt_att @ -E_w_est
        rc_neg = gt_att @ -E_w_flip_est

        so3_pos[:, 3] = rc_pos
        so3_neg[:, 3] = rc_neg

        P_mc_pos = K @ so3_pos
        P_mc_neg = K @ so3_neg

        # chirality test here
        curr_crater_center_homo = np.array([CW_params_sub[cc_id, 0:3]])
        curr_crater_center_homo = np.append(curr_crater_center_homo, 1)

        proj_pos = P_mc_pos @ curr_crater_center_homo.T
        proj_neg = P_mc_neg @ curr_crater_center_homo.T

        if proj_pos[2] > 0:  # chiraility test
            P_mc = P_mc_pos
            so3 = so3_pos
            cam_pos = E_w_est
        elif proj_neg[2] > 0:
            P_mc = P_mc_neg
            so3 = so3_neg
            cam_pos = E_w_flip_est
        else:
            continue

        ##################### extract new craters
        neighbouring_craters_id = np.arange(CW_params.shape[0])
        # 1) project all 3D points onto the image plane
        projected_3D_points = P_mc @ np.hstack(
            [CW_params[neighbouring_craters_id, 0:3], np.ones((len(neighbouring_craters_id), 1))]).T
        points_on_img_plane = np.array([projected_3D_points[0, :] / projected_3D_points[2, :],
                                        projected_3D_points[1, :] / projected_3D_points[2, :]])

        within_img_valid_indices = np.where((points_on_img_plane[0, :] >= 0) &
                                            (points_on_img_plane[0, :] <= img_w) &
                                            (points_on_img_plane[1, :] >= 0) &
                                            (points_on_img_plane[1, :] <= img_h) &
                                            ~np.isnan(points_on_img_plane[0, :]) &
                                            ~np.isnan(points_on_img_plane[1, :]))[0]

        fil_ncid = neighbouring_craters_id[within_img_valid_indices]

        # TODO: check if the crater is visible to the camera
        _, fil_ncid = visible_points_on_sphere(CW_params[:, 0:3], np.array([0, 0, 0]),
                                               np.linalg.norm(CW_params[0, 0:3]),
                                               cam_pos, fil_ncid)

        if len(fil_ncid) == 0:
            continue

        try:
            el_dist_mat = compute_ellipse_distance_matrix(CW_conic_inv, CW_Hmi_k, P_mc, CC_params,
                                                          fil_ncid)
        except:
            el_dist_mat = np.ones((CC_params.shape[0], len(neighbouring_craters_id))) * np.inf

        nearest_neighbors_idx, nearest_neighbors_val = find_nearest_neighbors(el_dist_mat)
        closest_neighbouring_ids = [fil_ncid[idx] for idx in nearest_neighbors_idx]

        # nn_val_stack.append(nearest_neighbors_val)
        matched_count = 0
        matched_ids = [[] for _ in range(CC_params.shape[0])]

        for j in range(CC_params.shape[0]):
            if nearest_neighbors_val[j] <= eld_thres:
                matched_ids[j] = ID[closest_neighbouring_ids[j]]
                matched_count = matched_count + 1
            else:
                matched_ids[j] = 'None'

        if matched_count > max_match_count:
            max_match_count = copy.deepcopy(matched_count)
            opt_matched_ids = copy.deepcopy(matched_ids)
            opt_cam_pos = copy.deepcopy(cam_pos)

    return max_match_count, opt_matched_ids, opt_cam_pos, nearest_neighbors_val


def visible_points_on_sphere(points, sphere_center, sphere_radius, camera_position, valid_indices):
    """Return the subset of the 3D points on the sphere that are visible to the camera."""
    visible_points = []
    visible_indices = []
    visible_len_P_cam = []

    for idx in valid_indices:
        point = points[idx, :]

        # 1. Translate the origin to the camera
        P_cam = point - camera_position

        # 2. Normalize the translated point
        P_normalized = P_cam / np.linalg.norm(P_cam)

        # 3 & 4. Solve for the real roots
        # Coefficients for the quadratic equation
        a = np.dot(P_normalized, P_normalized)
        b = 2 * np.dot(P_normalized, camera_position - sphere_center)
        c = np.dot(camera_position - sphere_center, camera_position - sphere_center) - sphere_radius ** 2

        discriminant = b ** 2 - 4 * a * c
        root1 = (-b + np.sqrt(discriminant)) / (2 * a)
        root2 = (-b - np.sqrt(discriminant)) / (2 * a)

        min_root = np.minimum(root1, root2)
        # 5. Check which real root matches the length of P_cam
        length_P_cam = np.linalg.norm(P_cam)

        # 6 & 7. Check visibility
        if (np.abs(min_root - length_P_cam) < 1000):
            visible_points.append(point)
            visible_indices.append(idx)
            visible_len_P_cam.append(length_P_cam)


    return visible_points, visible_indices

import argparse

def db_id_mask_from_sphere(CW_params, K, pos_est, att_est, pos_err, att_err):
    cam_view_vec = att_est[2, 0:3]
    # cam_view_vec = att_est[0:3, 2]
    moon_radius = 1737.4 * 1000
    
    f = K[0,0]
    c = K[0,2]
    max_img_ang = np.arctan((math.sqrt(c**2+c**2))/f)
    print("cam_view_vec",cam_view_vec)
    print("pos_est",pos_est)
    sphere_centre, sphere_radius, _ = bound_computation_sph_cylinder_inter_numba(cam_view_vec, K, att_est, pos_est, pos_err, att_err+max_img_ang, moon_radius)

    print(sphere_centre, sphere_radius)
    all_crater_coordinates = CW_params[:, 0:3]

    # Get all craters that lie in the intersection sphere.
    distances = np.linalg.norm(all_crater_coordinates-sphere_centre, axis = 1)
    inner_sphere_mask = distances <= sphere_radius
    return inner_sphere_mask

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to process data.")
    parser.add_argument("--data_dir", required=True, help="Directory path")
    parser.add_argument("--detections_dir", required=True, help="Directory path")
    parser.add_argument("--gt_data_dir", required=True, help="Directory path")
    parser.add_argument("--crater_catalogue_file", required=True, help="Crater catalogue path")
    parser.add_argument("--calibration_file", required=True, help="Crater calibration path")
    parser.add_argument("--flight_file", required=True, help="Flight path")
    parser.add_argument("--not_pangu",action='store_true') # extrinsic matrix will be calculated differently if the flight file was pangu generated or not pangu generated
    parser.add_argument("--silence_warnings",action='store_true')
    parser.add_argument("--lower_matched_percentage", required=True)
    parser.add_argument("--upper_matched_percentage", required=True)
    parser.add_argument("--write_dir", required=True)
    parser.add_argument("--catalogue_dir", required=True)
    parser.add_argument("--write_position_dir",required=True)
    parser.add_argument("--attitude_noise_deg", required=True)

    args = parser.parse_args()
    data_dir = args.data_dir
    detections_dir = args.detections_dir
    gt_data_dir = args.gt_data_dir
    result_dir = "result.txt"
    write_dir = args.write_dir
    catalogue_dir = args.catalogue_dir
    write_position_dir = args.write_position_dir
    attitude_noise_deg = args.attitude_noise_deg
    
    if args.silence_warnings:
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    calibration_file = args.calibration_file
    K, img_w, img_h = get_intrinsic(calibration_file)
    
    lower_matched_percentage = float(args.lower_matched_percentage)
    upper_matched_percentage = float(args.upper_matched_percentage)

    eld_thres=25
    # upper_matched_percentage=0.3 #.3 .6 TODO
    px_thres=5.0
    ab_thres=0.1
    deg_thres=10 #10
    # lower_matched_percentage=0.2 #.2 0.5
    
    starting_id = indx = 0#1575
    ending_id = 2500
    step = 500

    num_cores=4 #TODO: change to 4

    # ### Read the craters database in raw form
    all_craters_database_text_dir = args.crater_catalogue_file

    # Store large db as npz.
    cached_filename = catalogue_dir+"crater_catalogue.npz"
    # Get world crater info of the full Moon.
    print("Getting world craters.")
    CW_params_full, CW_conic_full, CW_conic_inv_full, CW_ENU_full, CW_Hmi_k_full, ID_full, crater_center_point_tree_full, CW_L_prime_full = \
            read_crater_database(all_craters_database_text_dir, cached_filename, overwrite_catalogue = True)

    # Get the camera extrinsic matrix and add noise.
    camera_extrinsic = get_extrinsic_from_flight_file(args.flight_file, args.not_pangu,attitude_noise_deg)

    crater_cam_filenames = get_files_in_dir(os.path.abspath(detections_dir),"txt")
    crater_cam_filenames.sort(key=lambda x: int(x[:-4]))

    crater_cam_filenames = crater_cam_filenames[:ending_id]#TODO: remove

    print("Getting imaged parameters.")
    imaged_params, craters_indices = get_imaged_craters_filter_id(detections_dir, crater_cam_filenames, True, 0.7)#, ID=ID_full) #NOTE: remove ID to not filter out ids


    for i in range(starting_id, ending_id, step):
        print("Processing file: ",i)

        # plot_ellipses_with_ids(imaged_params[i], craters_indices[i], imaged_params_gt[i], craters_indices_gt[i])

        cam = camera_extrinsic[i]
    #     noisy_att = noisy_cam_orientations[i]
        gt_pos = -cam[0:3, 0:3].T @ cam[0:3, 3]
        
        gt_att = cam[0:3, 0:3]

        gt_ids = craters_indices[i]

        curr_img_params = imaged_params[i]

        start_time = time.time()  ###################### start time ####################################

        # Generate a crater catalogue on the fly.
        pos_est, att_est = gt_pos, gt_att #TODO: change this to a real estimate
        pos_err = 6700
        att_err = 0.01
        

        # Find the crater indicies of craters in the camera's fov
        print("Getting local crater catalogue.")
        CW_visable_crater_mask = db_id_mask_from_sphere(CW_params_full, K, pos_est, att_est, pos_err, att_err*np.pi/180)
        # CW_visable_crater_mask = db_id_mask_with_errors(CW_params_full, K, pos_est, att_est, pos_err, att_err, 10)
        print("Length of catalogue:",np.sum(CW_visable_crater_mask))
        # print("monte:",np.sum(CW_visable_crater_mask))
        CW_params, CW_conic, CW_conic_inv, CW_ENU, CW_Hmi_k, ID, CW_L_prime = CW_params_full[CW_visable_crater_mask], CW_conic_full[CW_visable_crater_mask], CW_conic_inv_full[CW_visable_crater_mask], CW_ENU_full[CW_visable_crater_mask], CW_Hmi_k_full[CW_visable_crater_mask], ID_full[CW_visable_crater_mask], CW_L_prime_full[CW_visable_crater_mask]

        curr_craters_id = np.array(craters_indices[i])

        print("Missing ids from cat.:",np.setdiff1d(craters_indices[i], ID, assume_unique=True))
        
        CC_params = np.zeros([len(curr_img_params), 5])
        CC_a = np.zeros([len(curr_img_params)])
        CC_b = np.zeros([len(curr_img_params)])
        CC_conics = np.zeros([len(curr_img_params), 3, 3])
        sigma_sqr = np.zeros([len(curr_img_params)])
        matched_idx = np.zeros([len(curr_img_params)])
        matched_ids = [[] for _ in range(len(curr_img_params))]
        ncp_match_flag = False
        cp_match_flag = False
        # Convert curr_img_params to CC_conics and compute sigma_sqr
        for j, param in enumerate(curr_img_params):
            CC_params[j] = param
            CC_conics[j] = ellipse_to_conic_matrix(*param)

        Rw_c = gt_att.T
        # compute first Acam

        found_flag = False

        # catalogue_id_index = []
        # for k, idx in enumerate(craters_indices[i]):
        #     catalogue_id_index.append(ID.tolist().index(idx))

        print("Matching ...")
        for cc_id in range(CC_params.shape[0]):
            results = Parallel(n_jobs=num_cores)(
                delayed(main_func)(
                    db_cw_id, CW_params, CW_ENU, CW_L_prime, Rw_c, CC_params, K, cc_id, gt_att,
                    CW_conic_inv, CW_Hmi_k,
                    px_thres, ab_thres, deg_thres,
                    eld_thres, img_w, img_h) for db_cw_id in range(CW_params.shape[0])
            )
            opt_num_matches_vec = [opt_num_matches[0] for opt_num_matches in results]

            # if there is one that's larger than
            if (np.max(opt_num_matches_vec) > upper_matched_percentage * CC_params.shape[0]):
                found_flag = True
                opt_matched_ids = results[np.argmax(opt_num_matches_vec)][1]
                opt_cam_pos = results[np.argmax(opt_num_matches_vec)][2]
                opt_vec= results[np.argmax(opt_num_matches_vec)][3]
                break

            if found_flag:
                break

        if not(found_flag):
            for j in range(CC_params.shape[0]):
                matched_ids[j] = 'None'
            opt_cam_pos = np.array([0, 0, 0])
            matched_ids = log_result(matched_ids, curr_craters_id, result_dir, i)
            # print(opt_vec)
        else:
            matched_ids = log_result(opt_matched_ids, curr_craters_id,result_dir, i)



        # TODO: uncomment
        # if not os.path.exists(write_dir):
        #     os.makedirs(write_dir)
        # with open(write_dir+str(i)+".txt", "w") as file:
        #     file.write("ellipse: x_centre, y_centre, semi_major_axis, semi_minor_axis, rotation, id\n")
        #     for index, matched_id in enumerate(matched_ids):
        #         if matched_id != "None":
        #             x, y, a, b, theta = CC_params[index]
        #             ellipse_str = ", ".join(str(ellipse_param) for ellipse_param in CC_params[index])
        #             file.write(ellipse_str+", "+matched_id+"\n")
        #     file.close()

        # if not os.path.exists(write_position_dir):
        #     os.makedirs(write_position_dir)
        # with open(write_position_dir+str(i)+".txt", "w") as f:
        #     f.write(str(opt_cam_pos[0])+", "+str(opt_cam_pos[1])+", "+str(opt_cam_pos[2])+"\n")
        #     # f.close()



