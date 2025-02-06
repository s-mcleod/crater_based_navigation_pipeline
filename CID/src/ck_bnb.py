

import copy
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.get_data import *
from src.utils import *
# from src.metrics_ck import *
from scipy.linalg import eig


import numba
from numba import njit, prange, cuda
from scipy.spatial import cKDTree
import random
import csv
import time
from joblib import Parallel, delayed
# import ellipse_distance.erberly_ed as erb_ed
# import pyvista as pv
import queue
from numba.typed import List

import cProfile

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


def compare_vectors(vec1, vec2):
    # Perform element-wise comparison
    comparison_result = vec1 == vec2

    # Convert Boolean array to integer array (1 for True, 0 for False)
    int_result = comparison_result.astype(int)

    return int_result

@njit
def compute_distance_matrix(db_CW_conic_inv, db_CW_Hmi_k, P_mc, CC_params, neighbouring_craters_id):
    # Initialize the distance matrix
    pdist_mat = np.zeros((CC_params.shape[0], len(neighbouring_craters_id)))


    for ncid in range(len(neighbouring_craters_id)):
        curr_db_CW_conic_inv = db_CW_conic_inv[ncid]

        # Project it down
        curr_A = conic_from_crater_cpu(curr_db_CW_conic_inv, db_CW_Hmi_k[ncid], P_mc)

        # Extract xy first
        curr_A_params = extract_ellipse_parameters_from_conic(curr_A)

        for cc_id in range(CC_params.shape[0]):
            pdist_mat[cc_id, ncid] = euclidean_distance_cpu(np.array(curr_A_params[1:3]), CC_params[cc_id, 0:2])

    return pdist_mat


@njit
def compute_ellipse_distance_matrix(db_CW_conic_inv, db_CW_Hmi_k, P_mc, scaled_CC_params, neighbouring_craters_id):
    # Initialize the distance matrix
    pdist_mat = np.ones((scaled_CC_params.shape[0], len(neighbouring_craters_id))) * np.inf

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

        for cc_id in range(scaled_CC_params.shape[0]):
            scaled_curr_A_params = np.array(curr_A_params[1:])
            # scaled_curr_A_params[2] = scaled_curr_A_params[2] / CC_a[cc_id]
            # scaled_curr_A_params[3] = scaled_curr_A_params[3] / CC_b[cc_id]
            pdist_mat[cc_id, ncidx] = np.linalg.norm(scaled_curr_A_params - scaled_CC_params[cc_id])

    return pdist_mat

@cuda.jit
def compute_distance_matrix_cuda(db_CW_params, db_CW_conic_inv, db_CW_Hmi_k, P_mc, CC_params, pdist_mat):
    i, j = cuda.grid(2)

    if i < pdist_mat.shape[0] and j < pdist_mat.shape[1]:
        curr_db_CW_conic_inv = cuda.local.array((3, 3), dtype=float64)
        for x in range(3):
            for y in range(3):
                curr_db_CW_conic_inv[x, y] = db_CW_conic_inv[j, x, y]

        # Project it down
        curr_A = cuda.local.array((3, 3), dtype=float64)

        conic_from_crater(curr_db_CW_conic_inv, db_CW_Hmi_k[j], P_mc, curr_A)

        # Extract xy first
        _, x, y, _, _, _ = extract_ellipse_parameters_from_conic_gpu(curr_A)
        # _, x_neg, y_neg, _, _, _ = extract_ellipse_parameters_from_conic_gpu(curr_A_neg)

        # Compute distance with N imaged conics
        curr_pnp_dist = euclidean_distance((x, y), (CC_params[i, 0], CC_params[i, 1]))
        # curr_neg_pnp_dist = euclidean_distance((x_neg, y_neg), (CC_params[i, 0], CC_params[i, 1]))
        # pdist_mat[i, j] = min(curr_pos_pnp_dist, curr_neg_pnp_dist)  # Store just the minimum
        pdist_mat[i, j] = curr_pnp_dist


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

    V = eig(Acam, Bcam, left=True, right=False)[1]
    D = np.real(eig(Acam, Bcam)[0])

    sameValue, uniqueValue, uniqueIdx = differentiate_values(D)

    sigma_1 = uniqueValue
    sigma_2 = sameValue

    d1 = V[:, uniqueIdx].T
    d1 = d1 / np.linalg.norm(d1)
    k = np.sqrt(np.trace(np.linalg.inv(Acam)) - (1 / sigma_2) * np.trace(np.linalg.inv(Bcam)))

    delta_cam_est = k * d1
    delta_cam_est_flip = -k * d1

    delta_cam_world_est = Rw_c @ delta_cam_est.T
    E_w_est = curr_CW_param[0:3] + delta_cam_world_est

    delta_cam_world_flip_est = Rw_c @ delta_cam_est_flip.T
    E_w_flip_est = curr_CW_param[0:3] + delta_cam_world_flip_est

    return E_w_est, E_w_flip_est



@njit
def pre_eig_solver(CW_params, CW_ENU, Rw_c, CC_params, K, craters_id):
    curr_CW_param = CW_params

    eps = 1
    values = np.array([1 / curr_CW_param[3] ** 2, 1 / curr_CW_param[4] ** 2, 1 / eps ** 2])
    Aell = np.diag(values)

    # Aell = np.diag([1 / curr_CW_param[3] ** 2, 1 / curr_CW_param[4] ** 2, 1 / eps ** 2])

    Rw_ell = CW_ENU

    Re_cam = np.dot(Rw_c.T, Rw_ell)
    Acam = np.dot(np.dot(Re_cam, Aell), Re_cam.T)

    curr_conic = CC_params[craters_id]
    R_ellipse = np.zeros((2, 2))
    R_ellipse[0, 0] = np.cos(curr_conic[4])
    R_ellipse[0, 1] = -np.sin(curr_conic[4])
    R_ellipse[1, 0] = np.sin(curr_conic[4])
    R_ellipse[1, 1] = np.cos(curr_conic[4])
    R_ellipse = -R_ellipse
    Kc = np.dot(np.linalg.inv(K), np.array([curr_conic[0], curr_conic[1], 1])) * K[0, 0]
    Ec = np.array([0, 0, 0])

    Uc = np.append(R_ellipse[:, 0], 0)
    Vc = np.append(R_ellipse[:, 1], 0)
    Nc = np.array([0, 0, 1])

    M = np.outer(Uc, Uc) / curr_conic[2] ** 2 + np.outer(Vc, Vc) / curr_conic[3] ** 2

    # W = Nc / np.dot(Nc, (Kc - Ec))
    W = Nc / np.dot(Nc.astype(np.float64), (Kc - Ec))

    P = np.identity(3) - np.outer((Kc - Ec), W)

    Q = np.outer(W, W)

    Bcam = np.dot(P.T, np.dot(M, P)) - Q

    return Acam, Bcam

def eig_solver(Acam, Bcam):
    V = eig(Acam, Bcam, left=True, right=False)[1]
    D = np.real(eig(Acam, Bcam)[0])
    return V, D

@njit
def post_eig_solver(Acam, Bcam, V, D, curr_CW_param, Rw_c):
    sameValue, uniqueValue, uniqueIdx = differentiate_values_numba(D)

    sigma_1 = uniqueValue
    sigma_2 = sameValue

    d1 = V[:, uniqueIdx].T
    d1 = d1 / np.linalg.norm(d1)
    k = np.sqrt(np.trace(np.linalg.inv(Acam)) - (1 / sigma_2) * np.trace(np.linalg.inv(Bcam)))

    delta_cam_est = k * d1
    delta_cam_est_flip = -k * d1

    delta_cam_world_est = np.dot(Rw_c, delta_cam_est.T)
    E_w_est = curr_CW_param[0:3] + delta_cam_world_est

    delta_cam_world_flip_est = np.dot(Rw_c, delta_cam_est_flip.T)
    E_w_flip_est = curr_CW_param[0:3] + delta_cam_world_flip_est

    return E_w_est, E_w_flip_est

def p1e_solver_numba(CW_params, CW_ENU, Rw_c, CC_params, K, craters_id):
    Acam, Bcam = pre_eig_solver(CW_params, CW_ENU, Rw_c, CC_params, K, craters_id)
    V, D = eig_solver(Acam, Bcam)
    E_w_est, E_w_flip_est = post_eig_solver(Acam, Bcam, V, D, CW_params, Rw_c)
    return E_w_est, E_w_flip_est


def read_crater_database(craters_database_text_dir, to_be_removed_dir):
    with open(craters_database_text_dir, "r") as f:
        lines = f.readlines()[1:]  # ignore the first line
    lines = [i.split(',') for i in lines]
    lines = np.array(lines)

    ID = lines[:, 0]
    lines = np.float64(lines[:, 1:])

    # convert all to conics
    # db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k = get_craters_world_numba(lines)
    db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, db_L_prime = get_craters_world_numba_new(lines)

    # remove craters
    # Read the file and store IDs in a list
    if to_be_removed_dir is None:
        crater_center_point_tree = cKDTree(db_CW_params[:, 0:3])

        return db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, ID, crater_center_point_tree, db_L_prime
    else:
        with open(to_be_removed_dir, 'r') as file:
            removed_ids = [line.strip() for line in file.readlines()]

        # Find the indices of these IDs in the ID array
        removed_indices = np.where(np.isin(ID, removed_ids))[0]

        # Remove the craters with indices in removed_indices from your data arrays
        db_CW_params = np.delete(db_CW_params, removed_indices, axis=0)
        db_CW_conic = np.delete(db_CW_conic, removed_indices, axis=0)
        db_CW_conic_inv = np.delete(db_CW_conic_inv, removed_indices, axis=0)
        db_CW_ENU = np.delete(db_CW_ENU, removed_indices, axis=0)
        db_CW_Hmi_k = np.delete(db_CW_Hmi_k, removed_indices, axis=0)
        db_L_prime = np.delete(db_L_prime, removed_indices, axis=0)
        ID = np.delete(ID, removed_indices, axis=0)

        # Identify indices of repetitive elements in ID
        unique_ID, indices, counts = np.unique(ID, return_index=True, return_counts=True)
        removed_indices = np.setdiff1d(np.arange(ID.shape[0]), indices)

        # Remove rows from matrices
        db_CW_params = np.delete(db_CW_params, removed_indices, axis=0)
        db_CW_conic = np.delete(db_CW_conic, removed_indices, axis=0)
        db_CW_conic_inv = np.delete(db_CW_conic_inv, removed_indices, axis=0)
        db_CW_ENU = np.delete(db_CW_ENU, removed_indices, axis=0)
        db_CW_Hmi_k = np.delete(db_CW_Hmi_k, removed_indices, axis=0)
        db_L_prime = np.delete(db_L_prime, removed_indices, axis=0)
        ID = np.delete(ID, removed_indices, axis=0)

        crater_center_point_tree = cKDTree(db_CW_params[:, 0:3])

        return db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, ID, crater_center_point_tree, db_L_prime


def strip_symbols(s, symbols):
    for symbol in symbols:
        s = s.replace(symbol, '')
    return s


import csv
import numpy as np
from ast import literal_eval

def testing_data_reading_general(dir):
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


def testing_data_reading(dir):
    # Read the CSV file
    # Read the CSV file
    with open(dir, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Skip headers
    data = data[1:]
    camera_extrinsic = np.zeros([len(data), 3, 4])
    camera_pointing_angle = np.zeros(len(data))
    imaged_params = []
    noisy_imaged_params = []
    conic_indices = []

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

        # Extract Conic Indices
        curr_conic_indices = np.array(eval(row[4]))
        conic_indices.append(curr_conic_indices)

    return camera_extrinsic, camera_pointing_angle, imaged_params, noisy_imaged_params, conic_indices


def backproject_image_corners(K, width, height, depth=1):
    """Backproject the four corners of the image plane to 3D points in the camera's coordinate system."""
    corners = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [0, height, 1],
        [width, height, 1]
    ])
    inv_K = np.linalg.inv(K)
    return depth * (inv_K @ corners.T).T

def intersect_ray_sphere(ray_origin, ray_direction, sphere_center, sphere_radius):
    # Convert to numpy arrays
    O = np.array(ray_origin)
    D = np.array(ray_direction)
    C = np.array(sphere_center)
    r = sphere_radius

    # Calculate coefficients of the quadratic equation
    a = np.dot(D, D)
    OC = O - C
    b = 2 * np.dot(D, OC)
    c = np.dot(OC, OC) - r**2

    # Calculate discriminant
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        # No intersection
        return None
    elif discriminant == 0:
        # One intersection (tangent to the sphere)
        t = -b / (2 * a)
        # intersection_point = O + t * D
        return t
    else:
        # Two intersections
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)
        # intersection_point1 = O + t1 * D
        # intersection_point2 = O + t2 * D
        # return [intersection_point1, intersection_point2]
        return np.minimum(t1, t2)
    
def cam_axis_moon_intersection(point, camera_position):
    direction = point - camera_position
    direction = direction / np.linalg.norm(direction)
    sphere_center = np.array([0, 0, 0])
    sphere_radius = 1737400

    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, camera_position - sphere_center)
    c = np.dot(camera_position - sphere_center, camera_position - sphere_center) - sphere_radius ** 2

    discriminant = b ** 2 - 4 * a * c
    root1 = (-b + np.sqrt(discriminant)) / (2 * a)
    root2 = (-b - np.sqrt(discriminant)) / (2 * a)

    min_root = np.minimum(root1, root2)
    return min_root

def cam_axis_moon_intersection_new(point, off_point, sphere_center):
    direction = point - off_point
    direction = direction / np.linalg.norm(direction)
    # sphere_center = np.array([0, 0, 0]) - off_point
    sphere_radius = 1737400
    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, off_point - sphere_center)
    c = np.dot(off_point - sphere_center, off_point - sphere_center) - sphere_radius ** 2

    discriminant = b ** 2 - 4 * a * c
    root1 = (-b + np.sqrt(discriminant)) / (2 * a)
    root2 = (-b - np.sqrt(discriminant)) / (2 * a)

    min_root = np.minimum(root1, root2)
    return min_root


@njit
def gaussian_angle(Ai_params, Aj_params):
    xc_i, yc_i, a_i, b_i, phi_i = Ai_params
    xc_j, yc_j, a_j, b_j, phi_j = Aj_params

    y_i = np.array([xc_i, yc_i])
    y_j = np.array([xc_j, yc_j])

    Yi_phi = np.array([[np.cos(phi_i), -np.sin(phi_i)], [np.sin(phi_i), np.cos(phi_i)]])
    Yj_phi = np.array([[np.cos(phi_j), -np.sin(phi_j)], [np.sin(phi_j), np.cos(phi_j)]])

    Yi_len = np.array([[1/a_i**2, 0], [0, 1/ b_i **2]])
    Yj_len = np.array([[1 / a_j ** 2, 0], [0, 1 / b_j ** 2]])

    Yi_phi_t = np.transpose(Yi_phi)
    Yj_phi_t = np.transpose(Yj_phi)

    Yi = np.dot(Yi_phi, np.dot(Yi_len, Yi_phi_t))
    Yj = np.dot(Yj_phi, np.dot(Yj_len, Yj_phi_t))

    Yi_det = np.linalg.det(Yi)
    Yj_det = np.linalg.det(Yj)

    # Compute the difference between the vectors
    diff = y_i - y_j

    # Compute the sum of the matrices
    Y_sum = Yi + Yj

    # Invert the resulting matrix
    Y_inv = np.linalg.inv(Y_sum)

    # Compute the expression
    exp_part  = np.exp(-0.5 * diff.T @ Yi @ Y_inv @ Yj @ diff)

    front_part = (4 * np.sqrt(Yi_det * Yj_det)) / np.linalg.det(Y_sum)

    dGA = np.arccos(np.minimum(front_part * exp_part, 1))
    return dGA**2

def compute_pairwise_ga(CC_params, neighbouring_craters_id, db_CW_conic, db_CW_Hmi_k, curr_cam):
    pairwise_ga = np.zeros((CC_params.shape[0], len(neighbouring_craters_id)))

    for ncid in range(len(neighbouring_craters_id)):
        A = conic_from_crater_cpu_mod(db_CW_conic[neighbouring_craters_id[ncid]], db_CW_Hmi_k[neighbouring_craters_id[ncid]],
                                      curr_cam)  # project them onto the camera
        # convert A to ellipse parameters
        flag, x_c, y_c, a, b, phi = extract_ellipse_parameters_from_conic(A)
        # compute ga with all imaged conics
        for cc_id in range(CC_params.shape[0]):
            pairwise_ga[cc_id, ncid] = gaussian_angle(CC_params[cc_id],
                                                      [x_c, y_c, a, b, phi])  # measure pairwise GA

    return pairwise_ga


def visualize_points_and_camera(curr_cam, radius, corners_3D_world_coords, intersections):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Estimate the center and radius of the sphere
    center = np.array([0, 0, 0])

    # Create a meshgrid for the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    cam_pos = - curr_cam[0:3, 0:3].T @ curr_cam[0:3, 3]

    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], color='b')
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='c', alpha=0.3)

    # Visualize the camera's orientation using its rotation matrix
    rotation_matrix = curr_cam[0:3, 0:3].T
    scale_factor = radius/10  # Adjust this value to change the length of the orientation arrows
    ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
              rotation_matrix[0, 0] * scale_factor, rotation_matrix[1, 0] * scale_factor,
              rotation_matrix[2, 0] * scale_factor,
              color='g', label='Right Direction', arrow_length_ratio=0.1)
    ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
              rotation_matrix[0, 1] * scale_factor, rotation_matrix[1, 1] * scale_factor,
              rotation_matrix[2, 1] * scale_factor,
              color='y', label='Down Direction', arrow_length_ratio=0.1)
    ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
              rotation_matrix[0, 2] * scale_factor, rotation_matrix[1, 2] * scale_factor,
              rotation_matrix[2, 2] * scale_factor,
              color='r', label='Forward Direction', arrow_length_ratio=0.1)

    for c_id, corner in enumerate(corners_3D_world_coords):
        corner = corner - cam_pos
        corner = corner / np.linalg.norm(corner)
        curr_vec = corner * intersections[c_id]
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                  curr_vec[0], curr_vec[1], curr_vec[2],
                  color='m', arrow_length_ratio=0.1)

    # for corner in corners_3D_world_coords:
    #     # corner = corner + cam_pos
    #     ax.scatter(corner[0], corner[1], corner[2], 'bx')
        # corner = corner - cam_pos
        # plt.plot([cam_pos[0], corner[0]],
        #          [cam_pos[1], corner[1]],
        #          [cam_pos[2], corner[2]], 'g-')

    # Setting labels (optional)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    # Show the plot
    plt.show()


def log_result(matched_ids, gt_ids, opt_cam_pos, gt_pos, elapsed_time, result_dir, i):
    pos_err = np.linalg.norm(opt_cam_pos - gt_pos)

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
    TPR = TP / (TP + FN) if TP + FN > 0 else np.nan
    FPR = FP / (FP + TN) if FP + TN > 0 else np.nan
    FNR = FN / (TP + FN) if TP + FN > 0 else np.nan
    TNR = TN / (FP + TN) if FP + TN > 0 else np.nan

    matching_rate = TP / len([gt_id for gt_id in gt_ids if gt_id != 'None'])

    opt_cam_pos_str = ', '.join(['{:.2f}'.format(val) for val in opt_cam_pos])
    gt_pos_str = ', '.join(['{:.2f}'.format(val) for val in gt_pos])

    # Format the results in a single line
    result_str = ("Testing ID: {} | Matched IDs: {} | TP Rate: {:.2f} | FP Rate: {:.2f} | FN Rate: {:.2f} "
                  "| TNR Count: {} | TN Count: {} | MR: {:.2f} | Pos Error: {:.2f} | Est Pos: {} | GT Pos: {} | Time: {:.2f}\n").format(
        i, ', '.join(str(id) for id in matched_ids), TPR, FPR, FNR, TNR, TN, matching_rate, pos_err, opt_cam_pos_str, gt_pos_str,
        elapsed_time)

    # Open the file in append mode and write the result
    with open(result_dir, 'a') as file:
        file.write(result_str)



# def log_result(matched_ids, gt_ids, opt_cam_pos, gt_pos, elapsed_time, result_dir):
#     # Convert arrays to sets for efficient operations
#     # Create dictionaries to map ids to their positions
#     # matched_ids_pos = {id: idx for idx, id in enumerate(matched_ids)}
#     gt_ids_pos = {id: idx for idx, id in enumerate(gt_ids)}
#
#     pos_err = np.linalg.norm(opt_cam_pos - gt_pos)
#
#     TP = 0
#     FP = 0
#     FN = 0
#     TN = 0
#
#     # Compute TP and FP
#     for idx, m in enumerate(matched_ids):
#         if m == 'None':  # Check if the string is 'None'
#             FN += 1
#         elif m in gt_ids_pos and gt_ids_pos[m] == idx:  # Correct match in the same position
#             TP += 1
#         else:
#             FP += 1
#
#     # Compute TPR
#     TPR = TP / len(gt_ids)
#     FPR = FP / len(gt_ids)
#     FNR = FN / len(gt_ids)
#
#     opt_cam_pos_str = ', '.join(['{:.2f}'.format(val) for val in opt_cam_pos])
#     gt_pos_str = ', '.join(['{:.2f}'.format(val) for val in gt_pos])
#
#     # save matched_ids
#     # Format the results in a single line
#     result_str = "Testing ID: {} | Matched IDs: {} | TP Rate: {:.2f} | FP Rate: {:.2f} | FN Rate: {:.2f} | Pos Error: {:.2f} | Est Pos: {} | GT Pos: {} | Time: {:.2f}\n".format(
#         str(i),
#         ', '.join(map(str,
#                       matched_ids)),
#         TPR, FPR, FNR, pos_err, opt_cam_pos_str, gt_pos_str, elapsed_time)
#
#     # Open the file in append mode and write the result
#     with open(result_dir, 'a') as file:
#         file.write(result_str)

# def nearest_neighbour(dist_matrix):
#     M, N = dist_matrix.shape
#     # Placeholder for nearest neighbors for each row
#     nearest_neighbors_id = -np.ones(M, dtype=np.int32)
#     nearest_neighbors_val = -np.ones(M, dtype=np.int32)
#
#     # Get the indices of the flattened distance matrix sorted by distance
#     sorted_indices = np.argsort(dist_matrix.ravel())
#
#     for index in sorted_indices:
#         i, j = np.unravel_index(index, (M, N))
#
#         # If there are more rows than columns, a column can be chosen multiple times.
#         # Otherwise, a column should be chosen at most once.
#         if nearest_neighbors_id[i] == -1 and (M > N or (j not in nearest_neighbors_id)):
#             nearest_neighbors_id[i] = j
#             nearest_neighbors_val[i] = dist_matrix[i, j]
#
#         # Break when all rows have been assigned
#         if np.all(nearest_neighbors_id != -1):
#             break
#
#     return nearest_neighbors_id, nearest_neighbors_val


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


def main_func(db_cw_id, CW_params, CW_ENU, CW_L_prime, Rw_c, CC_params, K, cc_id, gt_att,
              CW_conic_inv, CW_Hmi_k,
              px_thres, ab_thres, rad_thres,
              scaled_CC_params,
              eld_thres, img_w, img_h):
    # for db_cw_id in range(CW_params.shape[0]):
        # db_cw_id = 20934
    opt_num_matches = 0
    opt_cam_pos = np.array([0, 0, 0])
    opt_matched_ids = np.zeros(scaled_CC_params.shape[0])

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
        return opt_num_matches, opt_matched_ids, opt_cam_pos
        # continue

    # compute the distance here
    legit_flag, curr_A = conic_from_crater_cpu(CW_conic_inv[db_cw_id], CW_Hmi_k[db_cw_id], P_mc)
    # Extract xy first
    if not (legit_flag):
        return opt_num_matches, opt_matched_ids, opt_cam_pos
        # continue

    curr_A_params = extract_ellipse_parameters_from_conic(curr_A)

    nextStageFlag = False
    if curr_A_params[0]:
        px_dev = np.linalg.norm((curr_A_params[1:3]) - (CC_params[cc_id, 0:2]))
        a_dev = np.abs(curr_A_params[3] - CC_params[cc_id, 2]) / CC_params[cc_id, 2]
        b_dev = np.abs(curr_A_params[4] - CC_params[cc_id, 3]) / CC_params[cc_id, 3]
        phi_dev = np.abs(curr_A_params[-1] - CC_params[cc_id, -1])

        if px_dev < px_thres and a_dev < ab_thres and b_dev < ab_thres and phi_dev < np.radians(rad_thres):
            nextStageFlag = True

    if not (nextStageFlag):
        return opt_num_matches, opt_matched_ids, opt_cam_pos
        # continue

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

    # check if GT is in it
    if len(fil_ncid) == 0:
        return opt_num_matches, opt_matched_ids, opt_cam_pos
        # continue

    try:
        el_dist_mat = compute_ellipse_distance_matrix(CW_conic_inv, CW_Hmi_k, P_mc, scaled_CC_params,
                                                      fil_ncid)
    except:
        el_dist_mat = np.ones((scaled_CC_params.shape[0], len(neighbouring_craters_id))) * np.inf

    nearest_neighbors_idx, nearest_neighbors_val = find_nearest_neighbors(el_dist_mat)
    closest_neighbouring_ids = [fil_ncid[idx] for idx in nearest_neighbors_idx]

    # TODO: first level test, if it passes, go to second level
    matched_count = np.sum(nearest_neighbors_val <= eld_thres)
    if not (matched_count > lower_matched_percentage * CC_params.shape[0]):
        return opt_num_matches, opt_matched_ids, opt_cam_pos
        # continue

    # TODO: Level 2: implement another check here with the correspondence
    len_CC_params = CC_params.shape[0]
    CW_matched_ids = []
    CW_params_sub = np.zeros([len_CC_params, CW_params.shape[1]])
    CW_ENU_sub = np.zeros([len_CC_params, CW_ENU.shape[1], CW_ENU.shape[2]])
    CW_L_prime_sub = np.zeros([len_CC_params, CW_L_prime.shape[1], CW_L_prime.shape[2]])
    # CW_conic_inv_sub = np.zeros([len_CC_params, CW_conic_inv.shape[1], CW_conic_inv.shape[2]])
    # CW_Hmi_k_sub = np.zeros([len_CC_params, CW_Hmi_k.shape[1], CW_Hmi_k.shape[2]])

    for j in range(CC_params.shape[0]):
        CW_matched_ids.append(ID[closest_neighbouring_ids[j]])
        CW_params_sub[j] = CW_params[closest_neighbouring_ids[j]]
        CW_ENU_sub[j] = CW_ENU[closest_neighbouring_ids[j]]
        CW_L_prime_sub[j] = CW_L_prime[closest_neighbouring_ids[j]]
        # CW_conic_inv_sub[j] = CW_conic_inv[closest_neighbouring_ids[j]]
        # CW_Hmi_k_sub[j] = CW_Hmi_k[closest_neighbouring_ids[j]]

    # IDEA: A small BnB here to get the best Camera position.
    opt_num_matches, opt_matched_ids, opt_cam_pos = optimum_matches(CW_params, CW_conic_inv, CW_Hmi_k, ID,
                                                                    CW_params_sub, CW_ENU_sub, CW_L_prime_sub,
                                                                    Rw_c, CC_params, scaled_CC_params, K,
                                                                    eld_thres,
                                                                    img_w, img_h)

    return opt_num_matches, opt_matched_ids, opt_cam_pos


def optimum_matches(CW_params, CW_conic_inv, CW_Hmi_k, ID,
                    CW_params_sub, CW_ENU_sub, CW_L_prime_sub,
                    Rw_c, CC_params, scaled_CC_params, K, eld_thres,
                    img_w, img_h):
    '''

    :param CW_params_sub: CW correspondence for CC_params
    :param CW_ENU_sub:
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
        ##### extract visible craters
        # corners_3D_camera_coords = backproject_image_corners(K, img_w, img_h, depth=86000)
        # corners_3D_world_coords = [(gt_att.T @ corner) + cam_pos for corner in corners_3D_camera_coords]
        #
        # # determine roots.
        # intersection = []
        # for corner_3D in corners_3D_world_coords:
        #     intersection.append(cam_axis_moon_intersection(corner_3D, cam_pos))
        #
        # max_dist = np.max(np.nanmax(np.array(intersection)))
        #
        # neighbouring_craters_id = crater_center_point_tree.query_ball_point(cam_pos, max_dist)  # qu
        # neighbouring_craters_id = np.sort(neighbouring_craters_id)
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
            el_dist_mat = compute_ellipse_distance_matrix(CW_conic_inv, CW_Hmi_k, P_mc, scaled_CC_params,
                                                          fil_ncid)
        except:
            el_dist_mat = np.ones((scaled_CC_params.shape[0], len(neighbouring_craters_id))) * np.inf

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
        # cam_pos_dist.append(np.linalg.norm(gt_pos - cam_pos))
        # matched_count_stack.append(matched_count)
        if matched_count > max_match_count:
            max_match_count = copy.deepcopy(matched_count)
            opt_matched_ids = copy.deepcopy(matched_ids)
            opt_cam_pos = copy.deepcopy(cam_pos)

    return max_match_count, opt_matched_ids, opt_cam_pos


def visible_points_on_sphere(points, sphere_center, sphere_radius, camera_position, valid_indices):
    """Return the subset of the 3D points on the sphere that are visible to the camera."""
    visible_points = []
    visible_indices = []
    visible_len_P_cam = []
    non_visible_len_P_cam = []

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
        else:
            # non_visible_points.append(point)
            non_visible_len_P_cam.append(length_P_cam)

    # 4) impose a check that we didnt eliminate points that are within the visible region because of a sub-optimal thresholding above
    #         # compute min and max distance for the visible_pts with the camera,
    #         # if there are other points that are within that range, raise a flag
    if len(non_visible_len_P_cam) > 0 and len(visible_len_P_cam) > 0:
        if np.min(np.array(non_visible_len_P_cam)) < np.max(np.array(visible_len_P_cam)):
            print('Something is wrong\n')

    return visible_points, visible_indices

# def find_indices(ID, curr_craters_id):
#     # Create a dictionary to map each string in ID to its index
#     id_dict = {value: idx for idx, value in enumerate(ID)}
    
#     # Find the indices for the curr_craters_id in the ID
#     indices = [id_dict[crater_id] for crater_id in curr_craters_id if crater_id in id_dict]
    
#     return indices

def find_indices(ID, curr_craters_id):
    # Create a dictionary to map each string in ID to its index
    id_dict = {value: idx for idx, value in enumerate(ID)}
    
    # Initialize the binary vector and indices list
    binary_vector = []
    indices = []
    
    # Populate the binary vector and indices list
    for crater_id in curr_craters_id:
        if crater_id in id_dict:
            binary_vector.append(1)
            indices.append(id_dict[crater_id])
        else:
            binary_vector.append(0)
            # indices.append(None)
    return indices, np.array(binary_vector)

from scipy.spatial.transform import Rotation as R

def generate_points_within_radius(center, radius, N):
    points = []
    for _ in range(N):
        while True:
            point = center + np.random.uniform(-radius, radius, size=3)
            if np.linalg.norm(point - center) <= radius:
                points.append(point)
                break
    return np.array(points)

def generate_points_on_radius(center, radius, N):
    points = []
    for _ in range(N):
        # Generate random angles for spherical coordinates
        phi = np.random.uniform(0, np.pi)  # Azimuthal angle
        theta = np.random.uniform(0, 2 * np.pi)  # Polar angle
        
        # Convert spherical coordinates to Cartesian coordinates
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        # Shift the point by the center
        point = center + np.array([x, y, z])
        points.append(point)
    
    return np.array(points)

def generate_cube_and_random_points(center, distance, N):
    """
    Generate the vertices of an 8-vertex cube centered at a given 3D point
    and N uniformly random points within the cube.

    Parameters:
    center (tuple): A tuple (x, y, z) representing the center of the cube.
    distance (float): Half the length of the cube's edges.
    N (int): The number of random points to generate within the cube.

    Returns:
    tuple: A tuple containing two elements:
        - vertices (np.ndarray): An array of shape (8, 3) with the coordinates of the cube's vertices.
        - random_points (np.ndarray): An array of shape (N, 3) with the coordinates of the random points within the cube.
    """
    cx, cy, cz = center
    d = distance

    # Generate the vertices of the cube
    vertices = np.array([
        [cx - d, cy - d, cz - d],
        [cx + d, cy - d, cz - d],
        [cx - d, cy + d, cz - d],
        [cx + d, cy + d, cz - d],
        [cx - d, cy - d, cz + d],
        [cx + d, cy - d, cz + d],
        [cx - d, cy + d, cz + d],
        [cx + d, cy + d, cz + d]
    ])

    # Generate N uniformly random points within the cube
    random_points = np.random.uniform(low=[cx - d, cy - d, cz - d], high=[cx + d, cy + d, cz + d], size=(N, 3))

    return vertices, random_points

def generate_rotations_within_uncertainty(base_rotation, angular_uncertainty, N):
    rotations = []
    axis_angle = []
    base_rot = R.from_matrix(base_rotation)
    for _ in range(N):
        while True:
            # Generate a random rotation vector within the angular uncertainty
            axis = np.random.normal(size=3)
            axis /= np.linalg.norm(axis)  # Normalize to get a unit vector
            angle = np.random.uniform(-angular_uncertainty, angular_uncertainty)
            perturbation = R.from_rotvec(axis * angle)
            
            # Apply the perturbation to the base rotation
            new_rotation = perturbation * base_rot
            rotations.append(new_rotation.as_matrix())
            axis_angle.append(new_rotation.as_rotvec())
            break
    return rotations, axis_angle

def generate_rotations_with_fixed_uncertainty(base_rotation, angular_uncertainty, N):
    rotations = []
    base_rot = R.from_matrix(base_rotation)
    
    for _ in range(N):
        # Generate a random unit vector for the axis
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)  # Normalize to get a unit vector
        
        # Use the fixed angular uncertainty as the rotation angle
        perturbation = R.from_rotvec(axis * angular_uncertainty)
        
        # Apply the perturbation to the base rotation
        new_rotation = perturbation * base_rot
        rotations.append(new_rotation.as_matrix())
    
    return rotations

def find_vector_on_plane(normal_vector):
    # Ensure the normal vector is a numpy array
    normal_vector = np.array(normal_vector)
    
    # Choose a vector that is not parallel to the normal vector
    if normal_vector[0] != 0:
        perpendicular_vector = np.array([0, 1, 0])
    else:
        perpendicular_vector = np.array([1, 0, 0])
    
    # Compute the cross product
    vector_on_plane = np.cross(normal_vector, perpendicular_vector)
    
    # If the result is a zero vector, choose another perpendicular vector
    if np.all(vector_on_plane == 0):
        if perpendicular_vector[0] != 0:
            perpendicular_vector = np.array([0, 0, 1])
        else:
            perpendicular_vector = np.array([1, 0, 0])
        vector_on_plane = np.cross(normal_vector, perpendicular_vector)
    
    # Normalize the vector (optional)
    vector_on_plane_unit = vector_on_plane / np.linalg.norm(vector_on_plane)
    
    return vector_on_plane, vector_on_plane_unit


def geodesic_distance(vector1, vector2, radius):
    # Ensure the vectors are numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    # Normalize the vectors to be unit vectors
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    
    # Compute the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)
    
    # Compute the angle between the two vectors
    angle = np.arccos(dot_product)
    
    # Compute the geodesic distance
    distance = radius * angle
    return distance

def intersection_point(P, o, a):
    P = np.array(P)
    o = np.array(o)
    a = np.array(a)
    
    # Calculate the magnitude of P (distance from the origin)
    d = np.linalg.norm(P)
    
    # Calculate the scalar parameter t
    t = (d - np.dot(P / d, o)) / np.dot(P / d, a)
    
    # Calculate the intersection point
    intersection = o + t * a
    
    return intersection


def rotation_distance(R1, R2):
    """
    Computes the geodesic distance between two rotations.

    Parameters:
    R1 (numpy.ndarray): First rotation matrix (3x3).
    R2 (numpy.ndarray): Second rotation matrix (3x3).

    Returns:
    float: The geodesic distance between the two rotations.
    """
    # Compute the relative rotation matrix
    R_rel = np.dot(R1.T, R2)
    
    # Convert the relative rotation matrix to a rotation object
    rotation = R.from_matrix(R_rel)
    
    # Compute the geodesic distance (angle of the relative rotation)
    angle = rotation.magnitude()
    
    return angle

# @njit
def extract_ellipse_parameters_from_conic(conic):
    A = conic[0, 0]
    B = conic[0, 1] * 2
    C = conic[1, 1]
    D = conic[0, 2] * 2
    F = conic[1, 2] * 2
    G = conic[2, 2]

    # Sanity test.
    denominator = B ** 2 - 4 * A * C
    if (B ** 2 - 4 * A * C >= 0) or (C * np.linalg.det(conic) >= 0):
        # print('Conic equation is not a nondegenerate ellipse')
        return False, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001

    #  Method from:
    #  https://en.wikipedia.org/wiki/Ellipse
    #  Convention in wikipedia:
    #   [ A B/2  D/2]
    #   [ B/2 C  E/2]
    #   [ D/2 E/2 F]]
    #  The following equations reexpresses wikipedia's formulae in Christian et
    #  al.'s convention.

    # Get centres.
    try:
        x_c = (2 * C * D - B * F) / denominator
        y_c = (2 * A * F - B * D) / denominator

        # Get semimajor and semiminor axes.
        KK = 2 * (A * F ** 2 + C * D ** 2 - B * D * F + (B ** 2 - 4 * A * C) * G)
        root = math.sqrt((A - C) ** 2 + B ** 2)
        a = -1 * math.sqrt(KK * ((A + C) + root)) / denominator
        b = -1 * math.sqrt(KK * ((A + C) - root)) / denominator

        if B != 0:
            # phi = math.atan((C - A - root) / B)  # Wikipedia had this as acot; should be atan. Check https://math.stackexchange.com/questions/1839510/how-to-get-the-correct-angle-of-the-ellipse-after-approximation/1840050#1840050
            # phi = math.atan2((C - A - root), B)  # Wikipedia had this as acot; should be atan. Check https://math.stackexchange.com/questions/1839510/how-to-get-the-correct-angle-of-the-ellipse-after-approximation/1840050#1840050
            phi = 0.5 * math.atan2(-B, (C - A)) - (-np.pi) # to convert to the positive realm
        elif A < C:
            phi = 0
        else:
            phi = math.pi / 2

        return True, x_c, y_c, a, b, phi
    except:
        return False, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001

@njit
def craterCS_to_camCS(C_conic_inv, Hmi_k, Pm_c):
    '''
    :param C_conic_inv: [3x3]
    :param Hmi_k: [4x3]
    :param Pm_c: [3x4]
    :param A: [3x3]
    :return:
    '''
    # print('yo')
    # Hci = np.dot(Pm_c, Hmi_k)
    Hci = matrix_multiply_cpu(Pm_c, Hmi_k, 3, 4, 3)
    # Astar = np.dot(np.dot(Hci, C_conic_inv), Hci.T)
    Astar = matrix_multiply_cpu(Hci, C_conic_inv, 3, 3, 3)
    Astar = matrix_multiply_cpu(Astar, Hci.T, 3, 3, 3)
    # A = np.linalg.inv(Astar)
    legit_flag, A = inverse_3x3_cpu(Astar)

    return legit_flag, A
    
def camCS_to_craterCS(Astar, Hmi_k, Pm_c):
    '''
    :param C_conic_inv: [3x3]
    :param Hmi_k: [4x3]
    :param Pm_c: [3x4]
    :param A: [3x3]
    :return:
    '''
    # print('yo')
    # Hci = np.dot(Pm_c, Hmi_k)
    Hci = matrix_multiply_cpu(Pm_c, Hmi_k, 3, 4, 3)
    Hci_inv = np.linalg.inv(Hci)
    Cstar =  Hci_inv @ Astar @ Hci_inv.T
    # C = np.linalg.inv(Cstar)
    # Astar = np.dot(np.dot(Hci, C_conic_inv), Hci.T)
    # Astar = matrix_multiply_cpu(Hci, C_conic_inv, 3, 3, 3)
    # Astar = matrix_multiply_cpu(Astar, Hci.T, 3, 3, 3)
    # # A = np.linalg.inv(Astar)
    # legit_flag, A = inverse_3x3_cpu(Astar)

    return Cstar


@njit
def matrix_multiply_cpu(A, B, A_rows, A_cols, B_cols):
    C = np.zeros((A_rows, B_cols))
    for i in range(A_rows):
        for j in range(B_cols):
            C[i, j] = 0.0
            for k in range(A_cols):
                C[i, j] += A[i, k] * B[k, j]

    return C


def get_points_on_ellipse(x, y, a, b, theta, N):
    """
    Returns N equidistant points on an ellipse centered at (x, y) with semi-major axis a,
    semi-minor axis b, and rotated by theta radians.
    
    Parameters:
    x (float): x-coordinate of the center of the ellipse
    y (float): y-coordinate of the center of the ellipse
    a (float): semi-major axis of the ellipse
    b (float): semi-minor axis of the ellipse
    theta (float): rotation angle of the ellipse in radians
    N (int): number of points to generate on the ellipse
    
    Returns:
    list of tuples: list of N (x, y) points on the ellipse
    """
    # Parametric equations for an ellipse centered at (0, 0)
    def parametric_ellipse(t, a, b):
        return a * np.cos(t), b * np.sin(t)
    
    # Rotation matrix for angle theta
    def rotate(x, y, theta):
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        return x_rot, y_rot
    
    # Generate N parametric angles
    t_values = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    points = []
    for t in t_values:
        # Get the points on the ellipse before rotation and translation
        point = parametric_ellipse(t, a, b)
        
        # Rotate the points
        point_rot = rotate(*point, theta)
        
        # Translate the points to the center (x, y)
        point_final = (point_rot[0] + x, point_rot[1] + y)
        
        points.append(point_final)
    
    return np.array(points)

def plot_ellipse_and_points(x, y, a, b, theta, point1, point2, point3):
    # Function to generate points on the ellipse
    def ellipse_points(x, y, a, b, theta, num_points=100):
        t = np.linspace(0, 2 * np.pi, num_points)
        ellipse_x = a * np.cos(t)
        ellipse_y = b * np.sin(t)
        
        # Rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        # Rotate and translate points
        ellipse_points = np.dot(R, np.array([ellipse_x, ellipse_y]))
        ellipse_x_rot = ellipse_points[0, :] + x
        ellipse_y_rot = ellipse_points[1, :] + y
        
        return ellipse_x_rot, ellipse_y_rot
    
    # Get ellipse points
    ellipse_x, ellipse_y = ellipse_points(x, y, a, b, theta)
    
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.plot(ellipse_x, ellipse_y, label='Ellipse')
    plt.scatter(*point1, color='red', zorder=5, label='Point 1')
    plt.scatter(*point2, color='blue', zorder=5, label='Point 2')
    plt.scatter(*point3, color='magenta', zorder=5, label='Point 3')
    
    plt.scatter(x, y, color='black', zorder=5, label='Center')
    
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ellipse with Two Points')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

def plane_normal(point1, point2, point3):
    # Convert points to numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)
    ref = np.array(point3)
    
    # Create vectors from the reference point to the two points
    v1 = p1 - ref
    v2 = p2 - ref
    
    # Compute the cross product of the vectors
    normal = np.cross(v1, v2)
    
    # Normalize the normal vector
    normal_length = np.linalg.norm(normal)
    if normal_length == 0:
        raise ValueError("The provided points are collinear or not distinct, resulting in a zero-length normal vector.")
    
    normal = normal / normal_length
    
    return normal

import argparse


def intersect_ray_sphere_stable(ray_origin, ray_direction, sphere_center, sphere_radius):
    # Convert to numpy arrays
    O = np.array(ray_origin)
    D = np.array(ray_direction)
    C = np.array(sphere_center)
    r = sphere_radius

    # Calculate coefficients of the quadratic equation
    a = np.dot(D, D)
    OC = O - C
    f = OC
    d = ray_direction

    b = 2 * np.dot(D, OC)
    c = np.dot(OC, OC) - r**2

    # Calculate discriminant
    # discriminant = b**2 - 4*a*c
    f_minus_fdd = f - np.dot(f, d) * d
    discriminant = 4 * np.dot(d, d) * (r**2 - np.dot(f_minus_fdd, f_minus_fdd))

    if discriminant < 0:
        # No intersection
        return None
    elif discriminant == 0:
        # One intersection (tangent to the sphere)
        t = -b / (2 * a)
        # intersection_point = O + t * D
        return t
    else:
        # Two intersections
        sqrt_discriminant = np.sqrt(discriminant)
        # t1 = (-b + sqrt_discriminant) / (2 * a)
        # t2 = (-b - sqrt_discriminant) / (2 * a)
        sign_b = np.sign(b)
        q = -0.5 * (b + sign_b * sqrt_discriminant)
        t1 = c / q
        t2 = q / a 

        # return [intersection_point1, intersection_point2]
        return np.minimum(t1, t2)
    

def get_N_points_from_ellipse_on_moon(imaged_ellipse, K, Rcam_in_world, tcam_in_world, N, moon_radius, curr_R_radius, fov_radius, unc_thres):
    points = get_points_on_ellipse(*imaged_ellipse, N)
    points_mcs = []
    los_mcs = []
    for i in range(points.shape[0]):
        curr_pt = points[i, :]
        # print(curr_pt)
        los = np.linalg.inv(K) @ np.hstack([curr_pt, 1])
        los = los / np.linalg.norm(los)
        # convert to moon's coordinate
        curr_los_mcs = Rcam_in_world @ los
        
        # get depth
        depth_est = intersect_ray_sphere(tcam_in_world, curr_los_mcs, np.array([0, 0, 0]), moon_radius)
        
        if depth_est == None:
            inter_flag = check_if_cone_intersects_moon(los, tcam_in_world, curr_R_radius, fov_radius, unc_thres)
            if not(inter_flag): # make sure the entire cone does not intersects the moon
                continue
            else:
                los_mcs.append(curr_los_mcs)
            # continue

        points_mcs.append((Rcam_in_world @ (depth_est * los)) + tcam_in_world)
        los_mcs.append(curr_los_mcs)

    return np.array(points_mcs), np.array(los_mcs)


def craterCentre_from_pts(imaged_ellipse, K, Rcam_in_world, tcam_in_world):
    pt1, pt2, pt3 = get_points_on_ellipse(*imaged_ellipse)
    # plot_ellipse_and_points(*curr_imaged_ellipse[j], pt1, pt2, pt3)
    # get depths for pt1 and pt2
    los1 = np.linalg.inv(K) @ np.hstack([pt1, 1])
    los1 = los1 / np.linalg.norm(los1)
    
    los2 = np.linalg.inv(K) @ np.hstack([pt2, 1])
    los2 = los2 / np.linalg.norm(los2)

    los3 = np.linalg.inv(K) @ np.hstack([pt3, 1])
    los3 = los3 / np.linalg.norm(los3)

    # convert to moon's coordinate
    curr_los1_mcs = Rcam_in_world @ los1
    curr_los2_mcs = Rcam_in_world @ los2
    curr_los3_mcs = Rcam_in_world @ los3
    
    depth_est1 = intersect_ray_sphere(tcam_in_world, curr_los1_mcs, np.array([0, 0, 0]), 1737400)
    depth_est2 = intersect_ray_sphere(tcam_in_world, curr_los2_mcs, np.array([0, 0, 0]), 1737400)
    depth_est3 = intersect_ray_sphere(tcam_in_world, curr_los3_mcs, np.array([0, 0, 0]), 1737400)

    pt1_mcs = (Rcam_in_world @ (depth_est1 * los1)) + tcam_in_world
    pt2_mcs = (Rcam_in_world @ (depth_est2 * los2)) + tcam_in_world
    pt3_mcs = (Rcam_in_world @ (depth_est3 * los3)) + tcam_in_world

    # get a plane
    plane = plane_normal(pt1_mcs, pt2_mcs, pt3_mcs)
    craterCentre = plane * 1737400
    return craterCentre

def delta_ab_from_camPos(craterCentre, Rworld_in_cam, tcam_in_world, Astar, ref_ellipse):
    # then backproject the conic, obtain a_u and b_u, then compute delta_a_u, delta_b_u
    virtual_T = np.zeros([3, 4])
    virtual_T[:, 0:3] = Rworld_in_cam
    virtual_T[:, 3] = Rworld_in_cam @ -tcam_in_world
    virtual_P = K @ virtual_T

    Hmi_k = craterCenter_to_Hmi_k(craterCentre)

    Cstar = camCS_to_craterCS(Astar, Hmi_k, virtual_P)
    _, *ellipse_param = extract_ellipse_parameters_from_conic(np.linalg.inv(Cstar))
    delta_a = np.abs(ellipse_param[2] - ref_ellipse[2])
    delta_b = np.abs(ellipse_param[3] - ref_ellipse[3])
    delta_theta = np.abs(ellipse_param[4] - ref_ellipse[4])
    return delta_a, delta_b, delta_theta

def craterCenter_to_Hmi_k(craterCentre):
    k = np.array([0, 0, 1])
    u = craterCentre / np.linalg.norm(craterCentre)
    e = np.cross(k, u) / np.linalg.norm(np.cross(k, u))
    n = np.cross(u, e) / np.linalg.norm(np.cross(u, e))

    TE_M = np.empty((3, 3), dtype=np.float64)
    TE_M[:, 0] = e
    TE_M[:, 1] = n
    TE_M[:, 2] = u

    # compute Hmi
    Hmi = np.hstack((TE_M.dot(S), craterCentre.reshape(-1, 1)))
    Hmi_k = np.vstack((Hmi, k.reshape(1, 3)))
    return Hmi_k

def plot_3d_points_with_residuals(points, residuals, tcam_in_world, delta_t, los_mcs, steepest_dir):
    """
    Plots a set of 3D points with a color scheme based on residuals.

    Parameters:
    points (array-like): A list or array of 3D points, shape (N, 3)
    residuals (array-like): A list or array of residuals, shape (N,)
    """
    points = np.array(points)
    residuals = np.array(residuals)
    point_down = -delta_t * tcam_in_world / np.linalg.norm(tcam_in_world)
    # cam_pointing_direction = delta_t * Rcam_in_world[:, 2]
    cam_pointing_direction = delta_t * los_mcs
    steepest_dir = delta_t * steepest_dir 

    if points.shape[1] != 3:
        raise ValueError("Points array must have a shape (N, 3)")
    if points.shape[0] != residuals.shape[0]:
        raise ValueError("Points array and residuals array must have the same length")
    
    # Create a scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with color mapping based on residuals
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=residuals, cmap='viridis')
    ax.quiver(*tcam_in_world, *point_down, color='r')
    ax.quiver(*tcam_in_world, *cam_pointing_direction, color='b')
    ax.quiver(*tcam_in_world, *steepest_dir, color='m')

    # Add a color bar to indicate the scale of residuals
    cbar = plt.colorbar(sc)
    cbar.set_label('Residuals')

    ##### plot the plane defined by the normal vector here
    # Unpack the normal vector and center point
    # a, b, c = Rcam_in_world[:, 2]
    a, b, c = los_mcs
    x0, y0, z0 = tcam_in_world

    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x0 - delta_t, x0 + delta_t, 10), np.linspace(y0 - delta_t, y0 + delta_t, 10))

    # Calculate the corresponding z values based on the plane equation
    zz = (-a * (xx - x0) - b * (yy - y0)) / c + z0

    # Plot the plane
    ax.plot_surface(xx, yy, zz, alpha=0.5, rstride=100, cstride=100)
    
    # Plot anchor points
    # ax.scatter(tcam_in_world[0], tcam_in_world[1], tcam_in_world[2], color='red', s=100, label='Anchor Point 1', edgecolor='k')
    # ax.scatter(0, 0, 0, color='blue', s=100, label='Anchor Point 2', edgecolor='k')
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    ax.set_title('3D Points with Residuals Color Scheme')
    
    # Show plot
    plt.show()


def steepest_direction(normal):
    """
    Computes the steepest direction on a plane defined by a normal vector.

    Parameters:
    normal (tuple): A tuple (a, b, c) representing the normal vector of the plane.

    Returns:
    tuple: A tuple representing the steepest direction in the plane.
    """
    a, b, c = normal

    # Choose a vector that is not parallel to the normal vector
    if a == 0 and b == 0:
        v = np.array([1, 0, 0])
    else:
        v = np.array([-b, a, 0])

    # Compute the cross product
    line_of_intersection = np.cross(normal, v)
    line_of_intersection = line_of_intersection / np.linalg.norm(line_of_intersection)
    w = np.cross(line_of_intersection, normal)

    # Normalize the direction vector
    w = w / np.linalg.norm(w)
    
    return w

def perturb_Rt(Rcam_in_world, tcam_in_world, delta_t, delta_rot):
    perturbed_tcam_in_world = generate_points_within_radius(tcam_in_world, delta_t, 1)
    perturbed_Rcam_in_world, perturbed_Rcam_in_world_axis_angle = generate_rotations_within_uncertainty(Rcam_in_world, delta_rot, 1)
    return perturbed_Rcam_in_world[0], perturbed_Rcam_in_world_axis_angle[0],  perturbed_tcam_in_world[0]


def visualise_plane_and_points(plane, points, proj_point):
    x_min = int(np.minimum(np.min(points[:, 0]) - 1000, proj_point[0] - 1000))
    x_max = int(np.maximum(np.max(points[:, 0]) + 1000, proj_point[0] + 1000))
    
    y_min = int(np.minimum(np.min(points[:, 1]) - 1000, proj_point[1] - 1000))
    y_max = int(np.maximum(np.max(points[:, 1]) + 1000, proj_point[1] + 1000))

    xx, yy = np.meshgrid(range(x_min, x_max), range(y_min, y_max))
    
    n = plane / np.linalg.norm(plane)
    # Calculate the corresponding z coordinates on the plane
    zz = (np.linalg.norm(plane) - n[0] * xx - n[1] * yy) / n[2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the plane
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='yellow')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r')
    ax.scatter(*proj_point, color='b')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.xlabel('z')
    plt.show()
    pass

def bound_computation(curr_point, K, Rc_cam_in_world, tc_cam_in_world, delta_t, delta_rot, moon_radius):
    los = np.linalg.inv(K) @ np.hstack([curr_point, 1])
    los = los / np.linalg.norm(los)
                
    # convert to moon's coordinate
    curr_los_mcs = (Rc_cam_in_world @ los)
    depth_est = intersect_ray_sphere(tc_cam_in_world, curr_los_mcs, np.array([0, 0, 0]), moon_radius)

    F = (moon_radius) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))
    E = intersection_point(F, tc_cam_in_world, curr_los_mcs)

    # convert to moon's coordinate
    E_sph = (Rc_cam_in_world @ (depth_est * los)) + tc_cam_in_world

    off_nadir_angle = np.arccos(np.dot(curr_los_mcs / np.linalg.norm(curr_los_mcs), -tc_cam_in_world / np.linalg.norm(tc_cam_in_world)))

    beta_p = off_nadir_angle + delta_rot
    beta_pp = delta_rot - off_nadir_angle

    if (np.abs(beta_p) >= np.pi/2) or (np.abs(beta_pp) >= np.pi/2):
        print('cannot be larger pi/2')

    h_p = delta_t / np.sin(beta_p)
    h_pp = delta_t / np.sin(beta_pp)

    h = np.linalg.norm(tc_cam_in_world) - moon_radius
    delta_d = h * np.tan(off_nadir_angle)
    # new implementation
    delta_x_1 = (h + h_p) * np.tan(beta_p) - delta_d
    delta_x_2 = (h + h_pp) * np.tan(beta_pp) + delta_d

    # convert to surface, get a vector on the surface.
    EF_dir = F - E
    EF_dir = EF_dir / np.linalg.norm(EF_dir)

    FE_dir = E - F
    FE_dir = FE_dir / np.linalg.norm(FE_dir)

    D = F + (delta_x_1 + delta_d) * FE_dir 
    G = F + (delta_x_2 - delta_d) * EF_dir
   
    B = (np.linalg.norm(tc_cam_in_world) + h_p) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))
    J = (np.linalg.norm(tc_cam_in_world) + h_pp) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))

    BD_dir = D - B 
    BD_dir = BD_dir / np.linalg.norm(BD_dir)
    JG_dir = G - J
    JG_dir = JG_dir / np.linalg.norm(JG_dir)

    BD_p_depth = intersect_ray_sphere(B, BD_dir, np.array([0, 0, 0]), moon_radius)
    JG_p_depth = intersect_ray_sphere(J, JG_dir, np.array([0, 0, 0]), moon_radius)

    # compute depth from J via GD_dir
    D_sph = B + BD_p_depth * BD_dir
    G_sph = J + JG_p_depth * JG_dir

    DE_dist = np.linalg.norm(D_sph - E_sph)
    GE_dist = np.linalg.norm(G_sph - E_sph)
    delta_x = np.maximum(DE_dist, GE_dist)
    return delta_x, np.maximum(delta_x_1, delta_x_2)

@njit
def rotation_matrix_from_axis_angle(angle, axis):
    """
    Compute the rotation matrix from an axis and an angle using Rodrigues' rotation formula.
    """
    kx, ky, kz = axis
    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])
    
    I = np.eye(3)
    K2 = np.dot(K, K)
    
    rotation_matrix = I + np.sin(angle) * K + (1 - np.cos(angle)) * K2
    
    return rotation_matrix


@njit
def get_rotation_matrix_axis_angle_numba(vec1, vec2):
    # Compute the cross product to get the axis of rotation
    axis = np.cross(vec1, vec2)
    
    # Compute the dot product to get the cosine of the angle
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Handle edge case for parallel vectors (avoid division by zero)
    if np.linalg.norm(axis) == 0:
        if cos_angle > 0:
            return np.eye(3)  # Identity matrix, no rotation needed
        else:
            # 180-degree rotation around any axis orthogonal to vec1
            # Choose an arbitrary orthogonal axis
            orthogonal_axis = np.array([1, 0, 0], dtype=np.float32)
            if vectors_are_close(vec1, orthogonal_axis) or vectors_are_close(vec1, -orthogonal_axis):
                orthogonal_axis = np.array([0, 1, 0], dtype=np.float32)
            return rotation_matrix_from_axis_angle(np.pi, orthogonal_axis)
    
    # Normalize the axis of rotation
    axis = axis / np.linalg.norm(axis)
    
    # Compute the angle of rotation
    angle = np.arccos(cos_angle)
    
    # Create the rotation matrix using the axis-angle representation
    rotation_matrix = rotation_matrix_from_axis_angle(angle, axis)
    
    return rotation_matrix

def get_rotation_matrix_axis_angle(vec1, vec2):
    # Compute the cross product to get the axis of rotation
    axis = np.cross(vec1, vec2)
    
    # Compute the dot product to get the cosine of the angle
    cos_angle = np.dot(vec1, vec2)
    
    # Handle edge case for parallel vectors (avoid division by zero)
    if np.linalg.norm(axis) == 0:
        if cos_angle > 0:
            return np.eye(3)  # Identity matrix, no rotation needed
        else:
            # 180-degree rotation around any axis orthogonal to vec1
            # Choose an arbitrary orthogonal axis
            orthogonal_axis = np.array([1, 0, 0])
            if np.allclose(vec1, orthogonal_axis) or np.allclose(vec1, -orthogonal_axis):
                orthogonal_axis = np.array([0, 1, 0])
            return R.from_rotvec(np.pi * orthogonal_axis).as_matrix()

    # Normalize the axis of rotation
    axis = axis / np.linalg.norm(axis)
    
    # Compute the angle of rotation
    angle = np.arccos(cos_angle)
    
    # Create the rotation using the axis-angle representation
    rotation = R.from_rotvec(angle * axis)
    
    # Extract the rotation matrix
    rotation_matrix = rotation.as_matrix()
    
    return rotation_matrix

def two_curves_one_curve_parameterization(r, phi, a, b):
    return np.array([r * np.cos(phi), r * np.sin(phi), np.sqrt(2 * a * (b + r * np.cos(phi)))]), \
            np.array([r * np.cos(phi), r * np.sin(phi), -np.sqrt(2 * a * (b + r * np.cos(phi)))])

def one_two_curves_ep_determination(r, phi, a, b):
    return np.array([r * np.cos(phi), r * np.sin(phi), np.sqrt(2 * a * (b + r * np.cos(phi)))])

@njit
def one_two_curves_ep_determination_numba(r, phi, a, b):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.sqrt(2 * a * (b + x))
    return np.array([x, y, z])


def extreme_point_determination(sph_radius, sph_offset, cylinder_radius):
    R = sph_radius
    a = sph_offset
    r = cylinder_radius

    # two curves, get only the top one
    # TODO: make sure we get the right side, probably can handle it properly with the rotation part, always rotate it to align with +z
    b = (R**2 - r**2 - a**2) / (2 * a)
    if R > (r + a): # two curves
        ep = np.zeros([4, 3])
        phi = np.radians(0)
        ep[0, :] = one_two_curves_ep_determination(r, phi, a, b)
        phi = np.radians(90)
        ep[1, :] = one_two_curves_ep_determination(r, phi, a, b)
        phi = np.radians(180)
        ep[2, :] = one_two_curves_ep_determination(r, phi, a, b)
        phi = np.radians(270)
        ep[3, :] = one_two_curves_ep_determination(r, phi, a, b)
    elif R < (r + a): # one curve
        phi_0 = np.arccos(-b /R)
        if phi_0 > np.radians(90):
            ep = np.zeros([5, 3])
            phi = np.radians(0)
            ep[0, :] = one_two_curves_ep_determination(r, phi, a, b)
            ep[1, :] = np.array([-b, np.sqrt(cylinder_radius**2 - b**2), 0])
            ep[2, :] = np.array([-b, -np.sqrt(cylinder_radius**2 - b**2), 0])
            phi = np.radians(90)
            ep[3, :] = one_two_curves_ep_determination(r, phi, a, b)
            phi = np.radians(270)
            ep[4, :] = one_two_curves_ep_determination(r, phi, a, b)
        else:
            ep = np.zeros([3, 3])
            phi = np.radians(0)
            ep[0, :] = one_two_curves_ep_determination(r, phi, a, b)
            ep[1, :] = np.array([-b, np.sqrt(cylinder_radius**2 - b**2), 0])
            ep[2, :] = np.array([-b, -np.sqrt(cylinder_radius**2 - b**2), 0])
        
    elif np.abs(R - (r + a)) < 1e-4: # self-intersecting
        ep = np.zeros([4, 3])
        phi = np.radians(0)
        ep[0, :] = one_two_curves_ep_determination(r, phi, a, b)
        phi = np.radians(90)
        ep[1, :] = one_two_curves_ep_determination(r, phi, a, b)
        phi = np.radians(180)
        ep[2, :] = one_two_curves_ep_determination(r, phi, a, b)
        phi = np.radians(270)
        ep[3, :] = one_two_curves_ep_determination(r, phi, a, b)
    else: #viviani's curve
        ep = np.zeros([4, 3])
        phi = np.radians(0)
        ep[0, :] = one_two_curves_ep_determination(r, phi, a, b)
        phi = np.radians(90)
        ep[1, :] = one_two_curves_ep_determination(r, phi, a, b)
        phi = np.radians(180)
        ep[2, :] = one_two_curves_ep_determination(r, phi, a, b)
        phi = np.radians(270)
        ep[3, :] = one_two_curves_ep_determination(r, phi, a, b)

    return np.array(ep)


@njit
def extreme_point_determination_numba(sph_radius, sph_offset, cylinder_radius):
    R = sph_radius
    a = sph_offset
    r = cylinder_radius

    b = (R**2 - r**2 - a**2) / (2 * a)
    
    if R > (r + a):  # two curves
        ep = np.zeros((4, 3), dtype=np.float64)
        phi = 0.0
        ep[0, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
        phi = np.pi / 2.0
        ep[1, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
        phi = np.pi
        ep[2, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
        phi = 3.0 * np.pi / 2.0
        ep[3, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
    elif R < (r + a):  # one curve
        phi_0 = np.arccos(-b / R)
        if phi_0 > np.pi / 2.0:
            ep = np.zeros((5, 3), dtype=np.float64)
            phi = 0.0
            ep[0, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
            ep[1, :] = np.array([-b, np.sqrt(cylinder_radius**2 - b**2), 0.0])
            ep[2, :] = np.array([-b, -np.sqrt(cylinder_radius**2 - b**2), 0.0])
            phi = np.pi / 2.0
            ep[3, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
            phi = 3.0 * np.pi / 2.0
            ep[4, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
        else:
            ep = np.zeros((3, 3), dtype=np.float64)
            phi = 0.0
            ep[0, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
            ep[1, :] = np.array([-b, np.sqrt(cylinder_radius**2 - b**2), 0.0])
            ep[2, :] = np.array([-b, -np.sqrt(cylinder_radius**2 - b**2), 0.0])
    elif np.abs(R - (r + a)) < 1e-4:  # self-intersecting
        ep = np.zeros((4, 3), dtype=np.float64)
        phi = 0.0
        ep[0, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
        phi = np.pi / 2.0
        ep[1, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
        phi = np.pi
        ep[2, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
        phi = 3.0 * np.pi / 2.0
        ep[3, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
    else:  # viviani's curve
        ep = np.zeros((4, 3), dtype=np.float64)
        phi = 0.0
        ep[0, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
        phi = np.pi / 2.0
        ep[1, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
        phi = np.pi
        ep[2, :] = one_two_curves_ep_determination_numba(r, phi, a, b)
        phi = 3.0 * np.pi / 2.0
        ep[3, :] = one_two_curves_ep_determination_numba(r, phi, a, b)

    return ep

def dist_to_midpoint(mid_pt, points):
    dists = np.zeros([points.shape[0]])
    for i in range(points.shape[0]):
        dists[i] = np.linalg.norm(points[i, :] - mid_pt)
    
    return dists, np.max(dists)

def enclosing_ball_determination(points):
    '''
    points: N x 3
    '''
    mid_pt = np.mean(points, axis=0)
    # find the largest distance
    radius, max_radius = dist_to_midpoint(mid_pt, points)
    return mid_pt, max_radius

@njit
def dist_to_midpoint_numba(mid_pt, points):
    dists = np.zeros(points.shape[0], dtype=np.float64)
    for i in range(points.shape[0]):
        dists[i] = np.linalg.norm(points[i, :] - mid_pt)
        # dists[i] = np.sqrt(np.sum(diff**2))
    
    return dists, np.max(dists)

@njit
def enclosing_ball_determination_numba(points):
    '''
    points: N x 3
    '''
    
    # mid_pt = np.mean(points, axis=0)
    mid_pt = mean_axis_0(points)
    # find the largest distance
    dists, max_radius = dist_to_midpoint_numba(mid_pt, points)
    return mid_pt, max_radius


@njit
def mean_axis_0(points):
    """
    Compute the mean along axis 0 for a 2D array.
    Equivalent to np.mean(points, axis=0).
    """
    num_points = points.shape[0]
    num_dimensions = points.shape[1]
    mean = np.zeros(num_dimensions, dtype=np.float64)
    
    for j in range(num_dimensions):
        sum_val = 0.0
        for i in range(num_points):
            sum_val += points[i, j]
        mean[j] = sum_val / num_points
    
    return mean

def intersects_xy_plane(origin, dir):
    length = -origin[2] / dir[2]
    
    # Calculate the intersection point
    intersection = origin + length * dir
    
    return intersection

@njit
def intersects_xy_plane_numba(origin, dir):
    # Handle division by zero in case dir[2] is zero
    if dir[2] == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)  # No intersection if direction is parallel to the plane
    
    length = -origin[2] / dir[2]
    
    # Calculate the intersection point
    intersection = origin + length * dir
    
    return intersection

@njit
def vectors_are_close(vec1, vec2, tol=1e-8):
    """
    Check if two vectors are element-wise close within a specified tolerance.
    """
    for i in range(len(vec1)):
        if abs(vec1[i] - vec2[i]) > tol:
            return False
    return True

def uncertainty_sphere_determination(tc_cam_in_world, x_bar, rot_axis, delta_rot, delta_t, moon_radius):
    R_ttokbar = R.from_rotvec(delta_rot * rot_axis).as_matrix()

    # generate k_bar.
    k_bar = R_ttokbar @ x_bar
    # now compute the rotation to align k_bar with -z
    R_kbartoz = get_rotation_matrix_axis_angle(k_bar, np.array([0, 0, -1]))
    # R_kbartoz_nb = get_rotation_matrix_axis_angle_numba(k_bar, np.array([0, 0, -1], dtype=np.float64))
    # then rotate kbar and t
    R_t = R_kbartoz @ tc_cam_in_world
    # R_t_nb = R_kbartoz_nb @ tc_cam_in_world

    # print(np.linalg.norm(R_t - R_t_nb))

    # then compute the intersection point (a) of rotated_kbar with the xz-plane
    a = intersects_xy_plane(R_t, np.array([0,0,1]))
    print("a",a)
    # a_nb = intersects_xy_plane_numba(R_t, np.array([0,0,1], dtype=np.float64))

    # print(np.linalg.norm(a - a_nb))

    # then compute the rotation to align a with the -x
    a_bar = a / np.linalg.norm(a)
    R_abartox = get_rotation_matrix_axis_angle(a_bar, np.array([-1, 0, 0]))
    R_a = R_abartox @ a

    # then compute the extreme points
    extreme_points = extreme_point_determination(moon_radius, np.linalg.norm(R_a), delta_t)
    extreme_points = extreme_points + R_a

    # extreme_points_nb = extreme_point_determination_numba(moon_radius, np.linalg.norm(R_a), delta_t)
    # extreme_points_nb = extreme_points_nb + R_a
    
    # ep_dists = []
    # for i in range(extreme_points.shape[0]):
    #     ep_dists.append(np.linalg.norm(extreme_points[i, :] - extreme_points_nb[i, :]))


    # then compute the MEB
    mid_pt, radius = enclosing_ball_determination(extreme_points)
    # mid_pt_nb, radius_nb = enclosing_ball_determination_numba(extreme_points)

    # print(np.linalg.norm(mid_pt - mid_pt_nb))
    # print(radius - radius_nb)
    
    # rotate it back
    R_all_inv = R_kbartoz.T @ R_abartox.T
    rotated_mid_pt = R_all_inv @ mid_pt
    
    return rotated_mid_pt, radius

@njit
def uncertainty_sphere_determination_numba(tc_cam_in_world, x_bar, rot_axis, delta_rot, delta_t, moon_radius):
    R_ttokbar = rotation_matrix_from_axis_angle(delta_rot, rot_axis)

    # generate k_bar.
    k_bar = R_ttokbar @ x_bar
    # now compute the rotation to align k_bar with -z
    # R_kbartoz = get_rotation_matrix_axis_angle(k_bar, np.array([0, 0, -1]))
    R_kbartoz = get_rotation_matrix_axis_angle_numba(k_bar, np.array([0, 0, -1], dtype=np.float64))
    # then rotate kbar and t
    R_t = R_kbartoz @ tc_cam_in_world
    # R_t_nb = R_kbartoz_nb @ tc_cam_in_world

    # print(np.linalg.norm(R_t - R_t_nb))

    # then compute the intersection point (a) of rotated_kbar with the xz-plane
    # a = intersects_xy_plane(R_t, np.array([0,0,1]))
    a = intersects_xy_plane_numba(R_t, np.array([0,0,1], dtype=np.float64))

    # print(np.linalg.norm(a - a_nb))

    # then compute the rotation to align a with the -x
    a_bar = a / np.linalg.norm(a)
    R_abartox = get_rotation_matrix_axis_angle_numba(a_bar, np.array([-1, 0, 0], dtype=np.float64))
    R_a = R_abartox @ a
    print("a",a)
    print("R_a",R_a)

    # then compute the extreme points
    if (np.linalg.norm(R_a) - delta_t > moon_radius ):
        print("*********No intersection**********")
        mid_pt = R_a/np.linalg.norm(R_a) * moon_radius
        # mid_pt = mid_pt + mid_pt
    else:
        extreme_points = extreme_point_determination_numba(moon_radius, np.linalg.norm(R_a), delta_t)
        extreme_points = extreme_points + R_a
        # extreme_points = extreme_points 
        print("extreme points\n",extreme_points)

        # extreme_points_nb = extreme_point_determination_numba(moon_radius, np.linalg.norm(R_a), delta_t)
        # extreme_points_nb = extreme_points_nb + R_a
        
        # ep_dists = []
        # for i in range(extreme_points.shape[0]):
        #     ep_dists.append(np.linalg.norm(extreme_points[i, :] - extreme_points_nb[i, :]))


        # then compute the MEB
        # mid_pt, radius = enclosing_ball_determination(extreme_points)
        mid_pt, radius = enclosing_ball_determination_numba(extreme_points)

    print("unrotated_mid_pt",mid_pt)

    # print(np.linalg.norm(mid_pt - mid_pt_nb))
    # print(radius - radius_nb)
    
    # rotate it back
    R_all_inv = R_kbartoz.T @ R_abartox.T
    rotated_mid_pt = R_all_inv @ mid_pt
    
    return rotated_mid_pt, radius, mid_pt

def bound_computation_sph_cylinder_inter(curr_los_mcs, K, Rc_cam_in_world, tc_cam_in_world, delta_t, delta_rot, moon_radius):
    # determine k_bar
    t_bar = tc_cam_in_world / np.linalg.norm(tc_cam_in_world)
    rot_axis = np.cross(-t_bar, curr_los_mcs)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    theta = np.arccos(np.dot(-t_bar, curr_los_mcs))

    # TODO: assert that beta is not larger than 90 degree, otherwise the cylinder does not intersect the sphere at all.
    if (delta_rot + theta) >= np.radians(90):
        return np.array([0,0,0]), 0, False
    
    ######## cylinder 1 #####
    mid_pt_1, r1 = uncertainty_sphere_determination(tc_cam_in_world, curr_los_mcs, rot_axis, delta_rot, delta_t, moon_radius)
    # mid_pt_1_nb, r1_nb = uncertainty_sphere_determination_numba(tc_cam_in_world, curr_los_mcs, rot_axis, delta_rot, delta_t, moon_radius)
    # print(np.linalg.norm(mid_pt_1_nb - mid_pt_1))
    # print(r1 - r1_nb)

    mid_pt_2, r2 = uncertainty_sphere_determination(tc_cam_in_world, curr_los_mcs, rot_axis, -delta_rot, delta_t, moon_radius)
    # mid_pt_2_nb, r2_nb = uncertainty_sphere_determination_numba(tc_cam_in_world, curr_los_mcs, rot_axis, -delta_rot, delta_t, moon_radius)
    # print(np.linalg.norm(mid_pt_2_nb - mid_pt_2))
    # print(r2 - r2_nb)
    mid_pt = (mid_pt_1 + mid_pt_2) / 2
    radius = np.linalg.norm(mid_pt - mid_pt_1)

    max_r = np.maximum(r1, r2)

    return mid_pt, max_r + radius, True

@njit
def bound_computation_sph_cylinder_inter_numba(curr_los_mcs, K, Rc_cam_in_world, tc_cam_in_world, delta_t, delta_rot, moon_radius):
    # determine k_bar
    t_bar = tc_cam_in_world / np.linalg.norm(tc_cam_in_world)
    rot_axis = np.cross(-t_bar, curr_los_mcs)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    theta = np.arccos(np.dot(-t_bar, curr_los_mcs))
    print("theta:",theta*180/np.pi)

    # TODO: assert that beta is not larger than 90 degree, otherwise the cylinder does not intersect the sphere at all.
    if (delta_rot + theta) >= (np.pi /2):
        print('something is wrong')
        return np.array([0,0,0], dtype=np.float64), 0.0, False
    
    ######## cylinder 1 #####
    # mid_pt_1, r1 = uncertainty_sphere_determination(tc_cam_in_world, curr_los_mcs, rot_axis, delta_rot, delta_t, moon_radius)
    mid_pt_1, r1, un_mid_pt_1 = uncertainty_sphere_determination_numba(tc_cam_in_world, curr_los_mcs, rot_axis, delta_rot, delta_t, moon_radius)
    print("mid_pt_1, r1",mid_pt_1, r1)

    # print(np.linalg.norm(mid_pt_1_nb - mid_pt_1))
    # print(r1 - r1_nb)

    # mid_pt_2, r2 = uncertainty_sphere_determination(tc_cam_in_world, curr_los_mcs, rot_axis, -delta_rot, delta_t, moon_radius)
    mid_pt_2, r2, un_mid_pt_2 = uncertainty_sphere_determination_numba(tc_cam_in_world, curr_los_mcs, rot_axis, -delta_rot, delta_t, moon_radius)
    print("mid_pt_2, r2",mid_pt_2, r2)

    um1 = un_mid_pt_1/np.linalg.norm(un_mid_pt_1)
    um2 = un_mid_pt_2/np.linalg.norm(un_mid_pt_2)
    ang_dist_unrot = np.arccos(np.dot(um1, um2))
    print("ang_dist_unrot",ang_dist_unrot)

    m1 = mid_pt_1/np.linalg.norm(mid_pt_1)
    m2 = mid_pt_2/np.linalg.norm(mid_pt_2)
    ang_dist_rot = np.arccos(np.dot(m1, m2))
    print("ang_dist_rot",ang_dist_rot)
    print("dist",(ang_dist_unrot-ang_dist_rot))

    # print(np.linalg.norm(mid_pt_2_nb - mid_pt_2))
    # print(r2 - r2_nb)
    mid_pt = (mid_pt_1 + mid_pt_2) / 2
    radius = np.linalg.norm(mid_pt - mid_pt_1)

    max_r = np.maximum(r1, r2)

    return mid_pt, max_r + radius, True


def bound_computation_v4(curr_point, K, Rc_cam_in_world, tc_cam_in_world, delta_t, delta_rot, moon_radius):
    los = np.linalg.inv(K) @ np.hstack([curr_point, 1])
    los = los / np.linalg.norm(los)
                
    # convert to moon's coordinate
    curr_los_mcs = (Rc_cam_in_world @ los)
    depth_est = intersect_ray_sphere(tc_cam_in_world, curr_los_mcs, np.array([0, 0, 0]), moon_radius)

    F = (moon_radius) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))
    E = intersection_point(F, tc_cam_in_world, curr_los_mcs)

    # convert to moon's coordinate
    E_sph = (Rc_cam_in_world @ (depth_est * los)) + tc_cam_in_world

    off_nadir_angle = np.arccos(np.dot(curr_los_mcs / np.linalg.norm(curr_los_mcs), -tc_cam_in_world / np.linalg.norm(tc_cam_in_world)))

    beta_p = off_nadir_angle + delta_rot
    beta_pp = delta_rot - off_nadir_angle

    if (np.abs(beta_p) >= np.pi/2) or (np.abs(beta_pp) >= np.pi/2):
        print('cannot be larger pi/2')

    h_p = delta_t / np.sin(beta_p)
    h_pp = delta_t / np.sin(beta_pp)

    h = np.linalg.norm(tc_cam_in_world) - moon_radius
    delta_d = h * np.tan(off_nadir_angle)
    # new implementation
    delta_x_1 = (h + h_p) * np.tan(beta_p) - delta_d


    # delta_x_2 = (h + h_pp) * np.tan(beta_pp) + delta_d

    # convert to surface, get a vector on the surface.
    EF_dir = F - E
    EF_dir = EF_dir / np.linalg.norm(EF_dir)

    FE_dir = E - F
    FE_dir = FE_dir / np.linalg.norm(FE_dir)

    D = F + (delta_x_1 + delta_d) * FE_dir 
    # G = F + (delta_x_2 - delta_d) * EF_dir
   
    B = (np.linalg.norm(tc_cam_in_world) + h_p) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))
    # J = (np.linalg.norm(tc_cam_in_world) + h_pp) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))

    BD_dir = D - B 
    BD_dir = BD_dir / np.linalg.norm(BD_dir)
    # JG_dir = G - J
    # JG_dir = JG_dir / np.linalg.norm(JG_dir)

    BD_p_depth = intersect_ray_sphere(B, BD_dir, np.array([0, 0, 0]), moon_radius)
    # JG_p_depth = intersect_ray_sphere(J, JG_dir, np.array([0, 0, 0]), moon_radius)

    # compute depth from J via GD_dir
    D_sph = B + BD_p_depth * BD_dir
    # G_sph = J + JG_p_depth * JG_dir

    DE_dist = np.linalg.norm(D_sph - E_sph)
    # GE_dist = np.linalg.norm(G_sph - E_sph)
    # delta_x = np.maximum(DE_dist, GE_dist)
    delta_x = DE_dist
    return delta_x, delta_x_1


def project_point_to_plane_3d_to_2d(P, P0, n, u, v):
    """
    Projects a 3D point onto a plane and returns its 2D coordinates on that plane.
    
    Parameters:
    P (tuple or list): The 3D point to be projected (x, y, z).
    P0 (tuple or list): A point on the plane (x0, y0, z0).
    u (tuple or list): The first basis vector of the plane (ux, uy, uz).
    v (tuple or list): The second basis vector of the plane (vx, vy, vz).
    
    Returns:
    tuple: The 2D coordinates (a, b) of the projected point on the plane.
    """
    # Convert input points to numpy arrays
    P = np.array(P)
    P0 = np.array(P0)
    u = np.array(u)
    v = np.array(v)
    
    w = P - P0
    d = np.dot(w, n)
    
    # P_proj is always behind the plane or on the plane, never above
    P_proj = P + d * n
    
    # Express the projected point in terms of the plane's basis vectors
    w_proj = P_proj - P0

    # TODO: visualise the plane, P_proj, P0, w_proj, d, n
    # visualize_projection(P, P0, u, v, n, P_proj, w_proj, rand_pts_on_plane)
    # Compute the coefficients directly using dot products
    a = np.dot(w_proj, u) / np.dot(u, u)
    b = np.dot(w_proj, v) / np.dot(v, v)
    
    # P_proj_plane_coord = a * u + b * v
    return P_proj, np.array([a, b])


def visualize_plane_and_spheres(plane_vector, sphere_params1, sphere_params2):
    # Extract plane parameters
    # a, b, c, d = plane_vector
    d = -np.linalg.norm(plane_vector)
    plane_vector = plane_vector / np.linalg.norm(plane_vector)


    # Create a meshgrid to define the plane with larger range
    x = np.linspace(-2000, 2000, 500)
    y = np.linspace(-2000, 2000, 500)
    x, y = np.meshgrid(x, y)
    z = (-plane_vector[0] * x - plane_vector[1] * y - d) / plane_vector[2]

    # Create a PyVista mesh for the plane
    plane_mesh = pv.StructuredGrid(x, y, z)

    # Create two spheres using the provided parameters
    sphere1 = pv.Sphere(center=(sphere_params1[0], sphere_params1[1], sphere_params1[2]), radius=sphere_params1[3])
    sphere2 = pv.Sphere(center=(sphere_params2[0], sphere_params2[1], sphere_params2[2]), radius=sphere_params2[3])

    # Plot the plane and spheres
    plotter = pv.Plotter()
    plotter.add_mesh(plane_mesh, color="red", opacity=0.5, label="Plane")
    plotter.add_mesh(sphere1, color="lightblue",  opacity=0.1, label="Sphere 1")
    plotter.add_mesh(sphere2, color="black", label="Sphere 2")
    plotter.add_legend()
    plotter.show()

def visualize_sphere_plane_points(plane_vector, sphere_params, points):
    # Extract plane parameters
    normal = np.array(plane_vector)
    distance = np.linalg.norm(normal)
    normal = normal / distance

    # Define the plane equation
    a, b, c = normal
    d = -distance

    # Create a meshgrid to define the plane with a larger range
    range_val = 200
    x = np.linspace(-range_val, range_val, 10)
    y = np.linspace(-range_val, range_val, 10)
    x, y = np.meshgrid(x, y)
    z = (-a * x - b * y - d) / c

    # Create a PyVista mesh for the plane
    plane_mesh = pv.StructuredGrid(x, y, z)

    # Create a sphere using the provided parameters
    sphere = pv.Sphere(center=(sphere_params[0], sphere_params[1], sphere_params[2]), radius=sphere_params[3])

    # Convert the points to a PyVista PolyData object
    points = np.array(points)
    point_cloud = pv.PolyData(points)

    # Plot the plane, sphere, and points
    plotter = pv.Plotter()
    plotter.add_mesh(plane_mesh, color="red", opacity=0.5, label="Plane")
    plotter.add_mesh(sphere, color="lightblue", opacity=1.0, label="Sphere")
    plotter.add_points(point_cloud, color="black", point_size=5, label="Points")
    plotter.add_legend()
    plotter.show()

def visualize_sphere_points(sphere_params, points):
    # Create a sphere using the provided parameters
    sphere = pv.Sphere(center=(sphere_params[0], sphere_params[1], sphere_params[2]), radius=sphere_params[3])

    # Convert the points to a PyVista PolyData object
    points = np.array(points)
    point_cloud = pv.PolyData(points)

    # Plot the plane, sphere, and points
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, color="lightblue", opacity=0.1, label="Sphere")
    plotter.add_points(point_cloud, color="black", point_size=5, label="Points")
    plotter.add_legend()
    plotter.show()


def bound_computation_v2(curr_point, K, Rc_cam_in_world, tc_cam_in_world, delta_t, delta_rot, moon_radius):
    los = np.linalg.inv(K) @ np.hstack([curr_point, 1])
    los = los / np.linalg.norm(los)
                
    # convert to moon's coordinate
    curr_los_mcs = (Rc_cam_in_world @ los)
    depth_est = intersect_ray_sphere(tc_cam_in_world, curr_los_mcs, np.array([0, 0, 0]), moon_radius)

    F = (moon_radius) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))
    E = intersection_point(F, tc_cam_in_world, curr_los_mcs)

    # convert to moon's coordinate
    E_sph = (Rc_cam_in_world @ (depth_est * los)) + tc_cam_in_world

    off_nadir_angle = np.arccos(np.dot(curr_los_mcs / np.linalg.norm(curr_los_mcs), -tc_cam_in_world / np.linalg.norm(tc_cam_in_world)))

    beta_p = off_nadir_angle + delta_rot
    beta_pp = delta_rot - off_nadir_angle

    if (np.abs(beta_p) >= np.pi/2) or (np.abs(beta_pp) >= np.pi/2):
        print('cannot be larger pi/2')

    h_p = delta_t / np.sin(beta_p)
    h_pp = delta_t / np.sin(beta_pp)

    h = np.linalg.norm(tc_cam_in_world) - moon_radius
    delta_d = h * np.tan(off_nadir_angle)
    # new implementation
    delta_x_1 = (h + h_p) * np.tan(beta_p) - delta_d
    delta_x_2 = (h + h_pp) * np.tan(beta_pp) + delta_d

    max_delta_x_on_plane = np.maximum(delta_x_1, delta_x_2)

    # convert to surface, get a vector on the surface.
    steep_dir = steepest_direction(tc_cam_in_world / np.linalg.norm(tc_cam_in_world))
    D = E + max_delta_x_on_plane * steep_dir
    G = E - max_delta_x_on_plane * steep_dir

    EF_dir = F - E
    EF_dir = EF_dir / np.linalg.norm(EF_dir)

    FE_dir = E - F
    FE_dir = FE_dir / np.linalg.norm(FE_dir)

    D = F + (delta_x_1 + delta_d) * FE_dir 
    G = F + (delta_x_2 - delta_d) * EF_dir
   
    B = (np.linalg.norm(tc_cam_in_world) + h_p) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))
    J = (np.linalg.norm(tc_cam_in_world) + h_pp) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))

    BD_dir = D - B 
    BD_dir = BD_dir / np.linalg.norm(BD_dir)
    JG_dir = G - J
    JG_dir = JG_dir / np.linalg.norm(JG_dir)

    BD_p_depth = intersect_ray_sphere(B, BD_dir, np.array([0, 0, 0]), moon_radius)
    JG_p_depth = intersect_ray_sphere(J, JG_dir, np.array([0, 0, 0]), moon_radius)

    # compute depth from J via GD_dir
    D_sph = B + BD_p_depth * BD_dir
    G_sph = J + JG_p_depth * JG_dir

    mid_point = (D_sph + G_sph) / 2
    mid_point_sph = (mid_point / np.linalg.norm(mid_point)) * moon_radius

    delta_x = np.linalg.norm(D_sph - mid_point_sph)

    # DE_dist = np.linalg.norm(D_sph - E_sph)
    # GE_dist = np.linalg.norm(G_sph - E_sph)
    # delta_x = np.maximum(DE_dist, GE_dist)
    return delta_x, np.maximum(delta_x_1, delta_x_2)


def bound_computation_v3(curr_point, K, Rc_cam_in_world, tc_cam_in_world, delta_t, delta_rot, moon_radius):
    los = np.linalg.inv(K) @ np.hstack([curr_point, 1])
    los = los / np.linalg.norm(los)
                
    # convert to moon's coordinate
    curr_los_mcs = (Rc_cam_in_world @ los)
    depth_est = intersect_ray_sphere(tc_cam_in_world, curr_los_mcs, np.array([0, 0, 0]), moon_radius)

    F = (moon_radius) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))
    E = intersection_point(F, tc_cam_in_world, curr_los_mcs)

    # convert to moon's coordinate
    E_sph = (Rc_cam_in_world @ (depth_est * los)) + tc_cam_in_world

    off_nadir_angle = np.arccos(np.dot(curr_los_mcs / np.linalg.norm(curr_los_mcs), -tc_cam_in_world / np.linalg.norm(tc_cam_in_world)))

    beta_p = off_nadir_angle + delta_rot
    beta_pp = delta_rot - off_nadir_angle

    if (np.abs(beta_p) >= np.pi/2) or (np.abs(beta_pp) >= np.pi/2):
        print('cannot be larger pi/2')

    h_p = delta_t / np.sin(beta_p)
    h_pp = delta_t / np.sin(beta_pp)

    h = np.linalg.norm(tc_cam_in_world) - moon_radius
    delta_d = h * np.tan(off_nadir_angle)
    # new implementation
    delta_x_1 = (h + h_p) * np.tan(beta_p) - delta_d
    delta_x_2 = (h + h_pp) * np.tan(beta_pp) + delta_d

    max_delta_x_on_plane = np.maximum(delta_x_1, delta_x_2)

    # convert to surface, get a vector on the surface.
    steep_dir = steepest_direction(tc_cam_in_world / np.linalg.norm(tc_cam_in_world))
    
    # check if steep_dir is similar direction to FE
    FE_dir = E - F
    FE_dir = FE_dir / np.linalg.norm(FE_dir)

    if np.arccos(np.dot(steep_dir, FE_dir)) > (np.pi / 2):
        steep_dir = -steep_dir
    
    H = E + max_delta_x_on_plane * steep_dir # a standardised notation for D and G

    # determine projection point
    if delta_x_1 > delta_x_2:
        # previously B
        vertex = (np.linalg.norm(tc_cam_in_world) + h_p) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))
    else:
        # previously J
        vertex = (np.linalg.norm(tc_cam_in_world) + h_pp) * (tc_cam_in_world / np.linalg.norm(tc_cam_in_world))

    H_vertex_dir = H - vertex
    H_vertex_dir = H_vertex_dir / np.linalg.norm(H_vertex_dir)
   
    HV_depth = intersect_ray_sphere(vertex, H_vertex_dir, np.array([0, 0, 0]), moon_radius)
   
    # compute depth from J via GD_dir
    H_sph = vertex + HV_depth * H_vertex_dir
    
    delta_x = np.linalg.norm(E_sph - H_sph)

    return delta_x, np.maximum(delta_x_1, delta_x_2)


def split_cube(center, r):
    # Unpack the center coordinates
    cx, cy, cz = center

    # Calculate the new radius for each sub-cube
    new_r = r / 2

    # Determine the offsets to calculate the centers of the sub-cubes
    offsets = [
        (dx, dy, dz)
        for dx in [-new_r, new_r]
        for dy in [-new_r, new_r]
        for dz in [-new_r, new_r]
    ]

    # Calculate the center points of the 8 sub-cubes
    sub_cube_centers = [
        (cx + dx, cy + dy, cz + dz) for dx, dy, dz in offsets
    ]

    # Each sub-cube will have the same radius
    sub_cube_radius = new_r

    return np.array(sub_cube_centers), sub_cube_radius

def point_ellipse_intersection(point, crater, enu, unc_thres):
    # project onto plane of each crater
    x_on_plane_proj, x_on_plane_proj_plane_coord = erb_ed.project_point_to_plane_3d_to_2d(point, 
                                                                                          crater[0:3], 
                                                                                          enu[:, 2], enu[:, 0], enu[:, 1])
    
    # x_on_plane_proj_nb, x_on_plane_proj_plane_coord_nb = erb_ed.project_point_to_plane_3d_to_2d_numba(np.array(point), 
    #                                                                                       np.array(crater[0:3]), 
    #                                                                                       np.array(enu[:, 2]), np.array(enu[:, 0]), np.array(enu[:, 1]))
    
    # print(np.linalg.norm(x_on_plane_proj - x_on_plane_proj_nb))
    # print(np.linalg.norm(x_on_plane_proj_plane_coord - x_on_plane_proj_plane_coord_nb))
    
    # then compute distance from ellipse
    x_on_ellipse = erb_ed.project_points_onto_ellipse(x_on_plane_proj_plane_coord[np.newaxis, 0:2], 
                                                            np.array([0, 0, crater[3], crater[4], crater[5]]))
    
    # x_on_ellipse_nb = erb_ed.project_points_onto_ellipse_numba(x_on_plane_proj_plane_coord[np.newaxis, 0:2], 
                                                            # np.array([0, 0, crater[3], crater[4], crater[5]]))
    
    # print(np.linalg.norm(x_on_ellipse_nb - x_on_ellipse))

    ellipse_dist = np.sqrt(np.linalg.norm(x_on_ellipse - x_on_plane_proj_plane_coord)**2 + np.linalg.norm(point - x_on_plane_proj)**2)
    if ellipse_dist <= unc_thres:
        return True
    else:
        return False

@njit
def point_ellipse_intersection_numba(point, crater, enu, unc_thres):
    # project onto plane of each crater using the Numba-compiled function
    x_on_plane_proj, x_on_plane_proj_plane_coord = erb_ed.project_point_to_plane_3d_to_2d_numba(
        point, 
        crater[0:3], 
        enu[:, 2], 
        enu[:, 0], 
        enu[:, 1]
    )
    
    # Compute distance from ellipse using the Numba-compiled function
    # Slice x_on_plane_proj_plane_coord_nb manually
    x_on_plane_proj_plane_coord_2d = np.zeros((1, 2))
    x_on_plane_proj_plane_coord_2d[0, :] = x_on_plane_proj_plane_coord[0:2]
    
    x_on_ellipse = erb_ed.project_points_onto_ellipse_numba(
        x_on_plane_proj_plane_coord_2d, 
        np.array([0, 0, crater[3], crater[4], crater[5]], dtype=np.float64)
    )
    
    ellipse_dist = np.sqrt(np.linalg.norm(x_on_ellipse - x_on_plane_proj_plane_coord_2d)**2 + np.linalg.norm(point - x_on_plane_proj)**2)
    return ellipse_dist <= unc_thres
    # return x_on_ellipse_nb

def obj_func(points_on_moon, crater_MCS, enu, unc_thres):
    # determine depth 
    obj_value = 0
    # matching_matrix = np.zeros([points_on_moon.shape[0], crater_MCS.shape[0]])
    for j in range(points_on_moon.shape[0]): # this could be parallelised
        for k in range(crater_MCS.shape[0]):
            curr_crater = crater_MCS[k, :]
            curr_enu = enu[k, :]
            matched = True
            for n in range(N_sampled_pts):
                curr_pts = points_on_moon[j, n, :]
                # evaluate if each k point intersects a crater
                intersect = point_ellipse_intersection(curr_pts, curr_crater, curr_enu, unc_thres)
                # intersect_nb = point_ellipse_intersection_numba(curr_pts, curr_crater, 
                #                                                 curr_enu, unc_thres)
                
                if not(intersect):
                    matched = False
                    break
            
            if matched:
                obj_value += 1
                # matching_matrix[j, k] = 1
                break
    
    return obj_value

@njit
def obj_func_numba(points_on_moon, crater_MCS, enu, unc_thres):
    #TODO: add a check to make sure the impropable geometry will not be evaluated

    # determine depth 
    obj_value = 0
    # matching_matrix = np.zeros([points_on_moon.shape[0], crater_MCS.shape[0]])
    for j in range(points_on_moon.shape[0]): # this could be parallelised
        for k in range(crater_MCS.shape[0]):
            curr_crater = crater_MCS[k, :]
            curr_enu = enu[k, :]
            matched = True
            for n in range(N_sampled_pts):
                curr_pts = points_on_moon[j, n, :]
                # evaluate if each k point intersects a crater
                # intersect = point_ellipse_intersection(curr_pts, curr_crater, curr_enu, unc_thres)
                intersect = point_ellipse_intersection_numba(curr_pts, curr_crater, 
                                                                curr_enu, unc_thres)
                
                if not(intersect):
                    matched = False
                    break
                
            if matched:
                obj_value += 1
                # matching_matrix[j, k] = 1
                break
    
    return obj_value

def upper_bound_func(los_mcs, crater_MCS, enu, unc_thres, K, Rc_cam_in_world, tc_cam_in_world, delta_t, delta_rot, moon_radius):
    obj_value = 0
    matching_matrix = np.zeros([los_mcs.shape[0], crater_MCS.shape[0]])
    for j in range(los_mcs.shape[0]): # this could be parallelised
        for k in range(crater_MCS.shape[0]):
            curr_crater = crater_MCS[k, :]
            curr_enu = enu[k, :]
            
            matched = True
            for n in range(N_sampled_pts):
                curr_los = los_mcs[j, n, :]
                # determine unc_thres_ub here
                unc_centre, unc_radius, legit_flag = bound_computation_sph_cylinder_inter(curr_los, K, 
                                                                                            Rc_cam_in_world, tc_cam_in_world, 
                                                                                            delta_t, delta_rot, moon_radius)
                # unc_centre_nb, unc_radius_nb, legit_flag = bound_computation_sph_cylinder_inter_numba(curr_los, K, 
                #                                                                             Rc_cam_in_world, tc_cam_in_world, 
                #                                                                             delta_t, delta_rot, moon_radius)
                
                # print(np.linalg.norm(unc_centre - unc_centre_nb))
                # print(unc_radius - unc_radius_nb)
                
                unc_radius = np.sqrt(unc_radius**2 + unc_radius**2)
                    
                # evaluate if each k point intersects a crater
                intersect = point_ellipse_intersection(unc_centre, curr_crater, curr_enu, unc_radius + unc_thres)
                if not(intersect):
                    matched = False
                    break
                
            if matched:
                obj_value += 1
                matching_matrix[j, k] = 1
                break
    
    return obj_value, matching_matrix

@njit
def upper_bound_func_numba(los_mcs, crater_MCS, enu, unc_thres, K, Rc_cam_in_world, tc_cam_in_world, delta_t, delta_rot, moon_radius):
    obj_value = 0
    # matching_matrix = np.zeros([los_mcs.shape[0], crater_MCS.shape[0]], dtype=np.float)
    for j in range(los_mcs.shape[0]): # this could be parallelised
        for k in range(crater_MCS.shape[0]):
            curr_crater = crater_MCS[k, :]
            curr_enu = enu[k, :]
            
            matched = True
            for n in range(N_sampled_pts):
                curr_los = los_mcs[j, n, :]
                # determine unc_thres_ub here
                # unc_centre, unc_radius, legit_flag = bound_computation_sph_cylinder_inter(curr_los, K, 
                #                                                                             Rc_cam_in_world, tc_cam_in_world, 
                #                                                                             delta_t, delta_rot, moon_radius)
                unc_centre, unc_radius, legit_flag = bound_computation_sph_cylinder_inter_numba(curr_los, K, 
                                                                                            Rc_cam_in_world, tc_cam_in_world, 
                                                                                            delta_t, delta_rot, moon_radius)
                
                # print(np.linalg.norm(unc_centre - unc_centre_nb))
                # print(unc_radius - unc_radius_nb)
                
                unc_radius = np.sqrt(unc_radius**2 + unc_radius**2)
                    
                # evaluate if each k point intersects a crater
                # intersect = point_ellipse_intersection(unc_centre, curr_crater, curr_enu, unc_radius + unc_thres)
                intersect = point_ellipse_intersection_numba(unc_centre, curr_crater, curr_enu, unc_radius + unc_thres)
                if not(intersect):
                    matched = False
                    break
                
            if matched:
                obj_value += 1
                # matching_matrix[j, k] = 1
                break
    
    return obj_value

def check_if_cone_intersects_moon(los, curr_tc_cam_in_world, rot_unc, fov_radius, unc_thres):
    rot_axis = np.cross(los, curr_tc_cam_in_world / np.linalg.norm(curr_tc_cam_in_world))
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    # rotate cam_principal_axis by positive and negative 
    # rot_unc = np.linalg.norm(curr_Rc_cam_in_world_axis_angle)
    total_rot_unc_for_cam = rot_unc + fov_radius + unc_thres

    cpa_R_pos = R.from_rotvec(total_rot_unc_for_cam * rot_axis).as_matrix()
    cpa_R_neg = R.from_rotvec(-total_rot_unc_for_cam * rot_axis).as_matrix()

    pos_vec = cpa_R_pos @ los
    neg_vec = cpa_R_neg @ los

    # check if they intersect the moon
    pos_depth = intersect_ray_sphere(curr_tc_cam_in_world, pos_vec, np.array([0,0,0]), moon_radius)
    neg_depth = intersect_ray_sphere(curr_tc_cam_in_world, neg_vec, np.array([0,0,0]), moon_radius)

    if (pos_depth is None) and (neg_depth is None):
        return False
    else:
        return True



def rotation_routine(R_q, curr_tc_cam_in_world, curr_t_radius, K, N_sampled_pts, moon_radius, crater_MCS, enu, unc_thres, lb, R_cube_thres, fov_radius):
    
    while not R_q.empty():
        R_priority, sub_R_cube = R_q.get()
        curr_Rc_cam_in_world_axis_angle = np.array(sub_R_cube[0:-1])
        curr_Rc_cam_in_world = R.from_rotvec(curr_Rc_cam_in_world_axis_angle).as_matrix()
        curr_R_radius = np.sqrt(2 * sub_R_cube_radius**2)

        # check camera principal axis here
        cam_principal_axis = curr_Rc_cam_in_world @ np.array([0,0,1])
        inter_flag = check_if_cone_intersects_moon(cam_principal_axis, curr_tc_cam_in_world, curr_R_radius, fov_radius, unc_thres)
        if not(inter_flag):
            continue

        points_on_moon = np.zeros([curr_imaged_ellipse.shape[0], N_sampled_pts, 3])
        los_mcs = np.zeros([curr_imaged_ellipse.shape[0], N_sampled_pts, 3])
        for j in range(curr_imaged_ellipse.shape[0]):
            points_on_moon[j, :, :], los_mcs[j, :, :], = get_N_points_from_ellipse_on_moon(curr_imaged_ellipse[j], K, 
                                                                        curr_Rc_cam_in_world, curr_tc_cam_in_world, 
                                                                        N_sampled_pts, moon_radius,
                                                                        curr_R_radius, fov_radius, unc_thres)
            
        obj_value = obj_func_numba(points_on_moon, crater_MCS, enu, unc_thres)

        if obj_value > lb:
            lb = obj_value
            opt_t = curr_tc_cam_in_world
            opt_R = curr_Rc_cam_in_world

        if not(k==0) and (-R_priority == obj_value):
            break

        print('inner iter:' + str(k) + ' lb:' +str(lb) + ' ub:' +str(-R_priority) + ' qsize:' + str(R_q.qsize()) + '\n')

        # Split R instead of t
        sub_R_cube_centers, sub_R_cube_radius = split_cube(curr_Rc_cam_in_world_axis_angle, sub_R_cube[-1])
        
        for j in range(sub_R_cube_centers.shape[0]):
            curr_Rc_cam_in_world_axis_angle = np.array(sub_R_cube_centers[j, :])
            curr_R_radius = np.sqrt(2 * sub_R_cube_radius**2)
            curr_Rc_cam_in_world = R.from_rotvec(curr_Rc_cam_in_world_axis_angle).as_matrix()

            ub_obj_value = upper_bound_func_numba(los_mcs, crater_MCS, enu, unc_thres, K, 
                                                    curr_Rc_cam_in_world, curr_tc_cam_in_world, curr_t_radius, curr_R_radius, moon_radius)
        
            if (ub_obj_value >= lb) and (sub_R_cube_radius > R_cube_thres):
                # add to the queue
                priority = -ub_obj_value
                element = (priority, (curr_Rc_cam_in_world_axis_angle[0], curr_Rc_cam_in_world_axis_angle[1], curr_Rc_cam_in_world_axis_angle[2], 
                                        sub_R_cube_radius))
                R_q.put(element)

        k += 1
    return lb, opt_t, opt_R


if __name__ == "__main__":
    data_dir = '/media/ckchng/1TBHDD/Dropbox/craters/global_DCID/data/'
    # data_dir = '/data/Dropbox/craters/global_DCID/data/'
    # /data/Dropbox/craters/global_DCID/data/testing_data_general_final_v1.csv
    ### Read the craters database in raw form
    all_craters_database_text_dir = data_dir + '/robbins_navigation_dataset_christians_all.txt'
    all_to_be_removed_dir = None

    CW_params, CW_conic, CW_conic_inv, CW_ENU, CW_Hmi_k, ID, crater_center_point_tree, CW_L_prime = \
            read_crater_database(all_craters_database_text_dir, all_to_be_removed_dir)

    calibration_file = data_dir + '/calibration.txt'
    K = get_intrinsic(calibration_file)

    # testing_data_dir = data_dir + '/exp1/testing_data_'+ str(noise_lvl) +'_60.csv'
    testing_data_dir = data_dir + '/testing_data_general_final_v1.csv'
    camera_extrinsic, camera_pointing_angle, imaged_params, noisy_imaged_params, craters_indices, \
    heights, noise_levels, remove_percentages, add_percentages, att_noises, noisy_cam_orientations = testing_data_reading_general(
        testing_data_dir)

    img_w = 1024
    img_h = 1024
    starting_id = 601
    ending_id = 602
    moon_radius = 1737400 / 1000
    delta_t = 0 # (km)
    delta_rot = np.deg2rad(5) # degree
    random.seed(10)
    unc_thres = 1e-1
    N_sampled_pts = 3
    
    R_cube_thres = np.deg2rad(0.01)
    t_cube_thres = 0.1
    fov_radius = np.deg2rad(30)
    
    log_dir = '/media/ckchng/1TBHDD/Dropbox/craters/global_DCID/output/' + str(starting_id) + '_' + str(ending_id) + '_harder.txt'
    setting_dir = '/media/ckchng/1TBHDD/Dropbox/craters/global_DCID/output/' + str(starting_id) + '_' + str(ending_id) + '_setting_harder.txt'
    
    # Define a priority queue
    t_q = queue.PriorityQueue()
    R_q = queue.PriorityQueue()
    
    # try just one
    for i in range(starting_id, ending_id):
        with open(log_dir, 'a') as f:
            f.write(str(i) + '\n')
        print(i)
        curr_noise_lvl = noise_levels[i]
        
        cam = camera_extrinsic[i]
        # get gt cam and gt_pos
        gt_pos = -cam[0:3, 0:3].T @ cam[0:3, 3]
        gt_att = cam[0:3, 0:3]

        Rworld_in_cam = gt_att
        Rcam_in_world = gt_att.T
        tcam_in_world = gt_pos
        tcam_in_world = tcam_in_world / 1000
        
        gt_ids = craters_indices[i]
        curr_imaged_ellipse = np.array(imaged_params[i])
        
        # TODO: get crater centers from gt_ids # later change this to sampling points on the crater rim
        curr_craters_id = np.array(craters_indices[i]) 
        # query the catalogue
        curr_craters_idx, flag = find_indices(ID, curr_craters_id)
        crater_MCS = CW_params[curr_craters_idx]
        crater_MCS[:, 0:-1] = crater_MCS[:, 0:-1]/1000
        enu = CW_ENU[curr_craters_idx, :, :]
        
        curr_imaged_ellipse = curr_imaged_ellipse[flag == 1]
        # filter curr_imaged_ellipse
        
        # TODO: then backproject the center point, get depth, then transform to the Moon's coordinate
        points_on_img_plane = curr_imaged_ellipse[:, 0:2].T
        
        los = np.zeros([points_on_img_plane.shape[1], 3])
        for pt_id in range(points_on_img_plane.shape[1]):
            curr_pt = points_on_img_plane[:, pt_id]
            los[pt_id, :] = np.linalg.inv(K) @ np.hstack([curr_pt, 1])
            
        # Get projection matrix
        T = cam
        P = K @ T

        Hmi_k = np.zeros((4, 3))

        ENU = np.zeros((3, 3))
        S = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        # Populate the matrices
        k = np.array([0, 0, 1])

        delta_a_u = np.zeros([points_on_img_plane.shape[1], 8])
        delta_b_u = np.zeros([points_on_img_plane.shape[1], 8])
        delta_a = np.zeros([points_on_img_plane.shape[1], 1000])
        delta_b = np.zeros([points_on_img_plane.shape[1], 1000])
        delta_theta = np.zeros([points_on_img_plane.shape[1], 1000])

        # create Rc and tc by perturbing it off the gt
        init_Rc_cam_in_world, init_Rc_cam_in_world_axis_angle, init_tc_cam_in_world  = perturb_Rt(Rcam_in_world, tcam_in_world, delta_t, delta_rot)

        element = (0, (init_tc_cam_in_world[0], init_tc_cam_in_world[1], init_tc_cam_in_world[2], delta_t * 2))
        t_q.put(element)

        element = (0, (init_Rc_cam_in_world_axis_angle[0], init_Rc_cam_in_world_axis_angle[1], init_Rc_cam_in_world_axis_angle[2], delta_rot * 2))
        R_q.put(element)
        
        lb = 0
        
        k = 0 # for debugging only
        # profiler = cProfile.Profile()
        # profiler.enable()

        while not t_q.empty():
            # evaluate obj
            t_priority, sub_t_cube = t_q.get()
            curr_tc_cam_in_world = np.array(sub_t_cube[0:-1])
            curr_t_radius = np.sqrt(2 * sub_t_cube[-1]**2)
            # rotation_routine
            curr_lb, _, inner_opt_R = rotation_routine(R_q, curr_tc_cam_in_world, 0, 
                                                            K, N_sampled_pts, moon_radius, crater_MCS, enu, unc_thres, lb,
                                                            fov_radius)

            if curr_lb > lb:
                lb = curr_lb
                opt_R = inner_opt_R
                opt_t = curr_tc_cam_in_world
                # TODO: run locally optimised routine
                # run locally optimise routine by iterating between t and R


            print('iter:' + str(k) + ' lb:' +str(lb) + ' ub:' +str(-t_priority) + ' qsize:' + str(t_q.qsize()) + '\n')
            
            if not(k==0) and (-t_priority == curr_lb):
                break

            sub_cube_centers, sub_cube_radius = split_cube(curr_tc_cam_in_world, sub_t_cube[-1])
            
            # TODO: parallelize here
            for j in range(sub_cube_centers.shape[0]):
                curr_tc_cam_in_world = np.array(sub_cube_centers[j, :])
                curr_t_radius = np.sqrt(2 * sub_cube_radius**2)

                ub, _, _ = rotation_routine(R_q, curr_tc_cam_in_world, curr_t_radius, 
                                                                K, N_sampled_pts, moon_radius, crater_MCS, enu, unc_thres, lb,
                                                                fov_radius)

                if ub > lb:
                    # add to the queue
                    priority = -ub
                    element = (priority, (curr_tc_cam_in_world[0], curr_tc_cam_in_world[1], curr_tc_cam_in_world[2], sub_cube_radius))
                    t_q.put(element)
                    # print('ub: ' + str(ub_obj_value) + '\n')
                    # print('domain added\n')


            # while not R_q.empty():
            #     R_priority, sub_R_cube = R_q.get()
            #     curr_Rc_cam_in_world_axis_angle = np.array(sub_R_cube[0:-1])
            #     curr_Rc_cam_in_world = R.from_rotvec(curr_Rc_cam_in_world_axis_angle).as_matrix()
            
            #     points_on_moon = np.zeros([curr_imaged_ellipse.shape[0], N_sampled_pts, 3])
            #     los_mcs = np.zeros([curr_imaged_ellipse.shape[0], N_sampled_pts, 3])
            #     for j in range(curr_imaged_ellipse.shape[0]):
            #         points_on_moon[j, :, :], los_mcs[j, :, :], = get_N_points_from_ellipse_on_moon(curr_imaged_ellipse[j], K, 
            #                                                                     curr_Rc_cam_in_world, curr_tc_cam_in_world, 
            #                                                                     N_sampled_pts, moon_radius)

            #     obj_value = obj_func_numba(points_on_moon, crater_MCS, enu, unc_thres)

            #     if obj_value > lb:
            #         lb = obj_value
            #         opt_t = curr_tc_cam_in_world
            #         opt_R = curr_Rc_cam_in_world

            #     if not(k==0) and (-R_priority == obj_value):
            #         break

            #     print('iter:' + str(k) + ' lb:' +str(lb) + ' ub:' +str(-R_priority) + ' qsize:' + str(R_q.qsize()) + '\n')

            #     # Split R instead of t
            #     sub_R_cube_centers, sub_R_cube_radius = split_cube(curr_Rc_cam_in_world_axis_angle, sub_R_cube[-1])
                
            #     for j in range(sub_R_cube_centers.shape[0]):
            #         curr_Rc_cam_in_world_axis_angle = np.array(sub_R_cube_centers[j, :])
            #         curr_R_radius = np.sqrt(2 * sub_R_cube_radius**2)
            #         curr_Rc_cam_in_world = R.from_rotvec(curr_Rc_cam_in_world_axis_angle).as_matrix()

            #         ub_obj_value = upper_bound_func_numba(los_mcs, crater_MCS, enu, unc_thres, K, 
            #                                                 curr_Rc_cam_in_world, curr_tc_cam_in_world, curr_t_radius, curr_R_radius, moon_radius)
                
            #         if (ub_obj_value >= lb) and (sub_R_cube_radius > R_cube_thres):
            #             # add to the queue
            #             priority = -ub_obj_value
            #             element = (priority, (curr_Rc_cam_in_world_axis_angle[0], curr_Rc_cam_in_world_axis_angle[1], curr_Rc_cam_in_world_axis_angle[2], 
            #                                   sub_R_cube_radius))
            #             R_q.put(element)

            #     k += 1

            
        print(lb)
        print(np.linalg.norm(tcam_in_world - opt_t))
        print(np.linalg.norm(Rcam_in_world - opt_R))
        print('ck')
                        

    
