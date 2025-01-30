import numpy as np
import cv2
import math
from sklearn.cluster import KMeans
from src.conics import *
import multiprocessing
import scipy.linalg

# Compares the position offset to height of camera above the surface.
# Returns the synthetic unit result and the corresponding real world units result.
def height_to_surface_metric(height_to_surface_synthetic_units, position_offset):
    return position_offset/height_to_surface_synthetic_units, height_to_surface_synthetic_units, position_offset

# Gets the widest edge of the projected camera image plane on the surface of the world. 
def observed_surface_width_metric(position_offset, K, k_extrinsic_matrix, image):
    ex = np.dot(np.linalg.inv(K), k_extrinsic_matrix)
    R = ex[0:3, 0:3]
    pos = -1*np.dot(np.linalg.inv(R), ex[0:3,3])
    # Image plane corner coordinates
    image_width, image_height, _ = image.shape
    pi = np.array([[0,0, 1],[0,image_width-1, 1],[image_height-1, image_width, 1],[image_height-1, 0, 1]])

    # Image points in camera frame.
    pc = [np.dot(np.linalg.inv(K), p) for p in pi]

    # World coordinates of image plane corners - these are the world coordinates of the frame in the 3d world.
    pw = [np.dot(np.transpose(R),p)+pos for p in pc]

    #Get vector of camera centre to image plane coordinates in the world reference frame.
    pv = [[p-pos] for p in pw]

    #Get 3D points on surface and height 0.
    surface_points = []
    for i in range(len(pv)):
        p_vec = pv[i][0]
        p_world = pw[i]
        t = -1*p_world[2]/p_vec[2]
        surface_points.append(np.array([p_world[0]+p_vec[0]*t, p_world[1]+p_vec[1]*t, 0]))

    # Get the lengths of the trapezium.
    side_lengths = [np.linalg.norm(surface_points[i-1]-surface_points[i]) for i in range(1, len(surface_points))]
    side_lengths.append(np.linalg.norm(surface_points[-1] - surface_points[0]))
    max_length = max(side_lengths)

    return position_offset/max_length, max_length, position_offset, surface_points

# Get the difference in estimated and ground truth rotation matricies (deg)
def angle_of_difference_rotation(K, k_extrinsic_matrix_estimated, k_extrinsic_matrix_ground_truth):
    k_extrinsic_matrix_estimated = np.dot(np.linalg.inv(K),k_extrinsic_matrix_estimated)
    k_extrinsic_matrix_ground_truth = np.dot(np.linalg.inv(K),k_extrinsic_matrix_ground_truth)
    R_est = k_extrinsic_matrix_estimated[0:3,0:3]
    R_gt = k_extrinsic_matrix_ground_truth[0:3,0:3]
    R = np.dot(R_gt, R_est.T)
    # TODO: we probably shouldn't be using min but sometimes we get 2.000000000004 which is just a rounding error.
    return math.acos(min((np.trace(R)-1),2)/2)

# Get the difference in estimated and ground truth position vectors (synthetic world units)
def get_position_offset(K, extrinsic_matrix_estimated, k_extrinsic_matrix_ground_truth, min_offset, scale):
    extrinsic_matrix_estimated = np.dot(np.linalg.inv(K),extrinsic_matrix_estimated)
    k_extrinsic_matrix_ground_truth = np.dot(np.linalg.inv(K),k_extrinsic_matrix_ground_truth)
    R_est = extrinsic_matrix_estimated[0:3,0:3]
    R_gt = k_extrinsic_matrix_ground_truth[0:3,0:3]

    estimated_world_position = -1*np.dot(np.linalg.inv(R_est), extrinsic_matrix_estimated[:,3])
    estimated_world_position = np.array([v*scale + min_offset for v in estimated_world_position])
    ground_truth_world_position = -1*np.dot(np.linalg.inv(R_gt), k_extrinsic_matrix_ground_truth[:,3])
    ground_truth_world_position = np.array([v*scale + min_offset for v in ground_truth_world_position])

    diff = np.linalg.norm(ground_truth_world_position-estimated_world_position)

    return diff, abs(ground_truth_world_position[0]-estimated_world_position[0]), abs(ground_truth_world_position[1]-estimated_world_position[1]), abs(ground_truth_world_position[2]-estimated_world_position[2])

def maass_pnp_reprojection_error(cubic_dual_craters, craters_world, Pm_c, K, weighted_matrix, continuous=False, Pm_c_true = np.identity(1)):
    W = np.diag(weighted_matrix)
    err = 0
    for i, c in enumerate(craters_world):
        crater_centre_pnp_error = maass_reprojection_error(cubic_dual_craters[i], c, Pm_c, K, continuous, Pm_c_true)
        err += W[i]*(crater_centre_pnp_error**2)
    return err

def maass_reprojection_error(cubic_dual_crater, crater_w, Pm_c, K, continuous=False, Pm_c_true=np.identity(1)):
    # Projected crater in world reference frame under estimated projection matrix.
    projected_crater_centre_w = crater_w.proj_crater_centre(Pm_c)

    # Dual crater projection.
    projected_crater_centre_c = cubic_dual_crater.proj_crater_centre(np.hstack((K,np.array([0,0,0]).reshape((3,1)))))
    
    reprojection_error = np.linalg.norm(projected_crater_centre_w-projected_crater_centre_c)
    
    return reprojection_error

def dual_pnp_reprojection_error(dual_craters, dual_crater_weights, craters_world, Pm_c, K, dual_method, weighted_matrix, continuous=False, Pm_c_true = np.identity(1)):
    W = np.diag(weighted_matrix)
    err = 0
    for i, c in enumerate(craters_world):
        crater_centre_pnp_error = dual_reprojection_error(dual_craters[i], c, dual_crater_weights[i], Pm_c, K, dual_method, continuous, Pm_c_true)
        err += W[i]*(crater_centre_pnp_error**2)
    return err

def dual_reprojection_error(dual_craters, crater_w, dual_scale, Pm_c, K, dual_method = "linear", continuous=False, Pm_c_true=np.identity(1)):
    # Projected crater in world reference frame under estimated projection matrix.
    projected_crater_centre_w = crater_w.proj_crater_centre(Pm_c)

    # Dual crater projection.
    projected_crater_centre_c1 = dual_craters[0].proj_crater_centre(np.hstack((K,np.array([0,0,0]).reshape((3,1)))))
    projected_crater_centre_c2 = dual_craters[1].proj_crater_centre(np.hstack((K,np.array([0,0,0]).reshape((3,1)))))
    
    # Cost.
    if dual_method == "sigmoid":
        reprojection_error = (sigmoid(dual_scale))*np.linalg.norm(projected_crater_centre_w-projected_crater_centre_c1) + (1 - sigmoid(dual_scale))*np.linalg.norm(projected_crater_centre_w-projected_crater_centre_c2)
    else:
        reprojection_error = linear(dual_scale)*np.linalg.norm(projected_crater_centre_w-projected_crater_centre_c1) + (1-linear(dual_scale))*np.linalg.norm(projected_crater_centre_w-projected_crater_centre_c2)
    
    return reprojection_error

# Takes value between 0 and 1.
def linear(s):
    return s

# Takes value in range -x to x.
def sigmoid(s):
    # return 4*(1/(1+math.e**(-w)))-2
    return 1/(1+math.e**(-s))

def get_weight(dual_method,s):
    if dual_method == "sigmoid":
        return sigmoid(s)
    return s

def crater_centre_pnp(A, crater, Pm_c, continuous=False, Pm_c_true=np.identity(1)):
    x_c, y_c = crater.proj_crater_centre(Pm_c)
    # If we are using continuous data, then we project the true crater centre (not entirely true of continuous data but we will go with it)
    if continuous:
        x_a, y_a = crater.proj_crater_centre(Pm_c_true)
        # x_a, y_a, _, _, _ = conic_matrix_to_ellipse(A)
    else:
        x_a, y_a, _, _, _ = conic_matrix_to_ellipse(A)
    euclid_dist = np.linalg.norm(np.array([x_a, y_a]) - np.array([x_c, y_c]))
    return euclid_dist

def crater_centre_pnp_metric(image_conics, craters_world, Pm_c, weighted_matrix, continuous=False, Pm_c_true = np.identity(1)):
    W = np.diag(weighted_matrix)
    err = 0
    for i, c in enumerate(craters_world):
        Ai = image_conics[i]
        crater_centre_pnp_error = crater_centre_pnp(Ai, c, Pm_c, continuous, Pm_c_true)
        err += W[i]*(crater_centre_pnp_error**2)
    return err

def gaussian_angle_ck(A, crater, Pm_c, continuous=False, Pm_c_true=np.identity(1)):

    C = conic_from_crater(crater, Pm_c)

    Ai = A
    Aj = C

    xc_i, yc_i, a_i, b_i, phi_i = conic_matrix_to_ellipse(Ai)
    xc_j, yc_j, a_j, b_j, phi_j = conic_matrix_to_ellipse(Aj)

    y_i = np.array([xc_i, yc_i])
    y_j = np.array([xc_j, yc_j])

    Yi_phi = np.array([[np.cos(phi_i), -np.sin(phi_i)], [np.sin(phi_i), np.cos(phi_i)]])
    Yj_phi = np.array([[np.cos(phi_j), -np.sin(phi_j)], [np.sin(phi_j), np.cos(phi_j)]])

    try:
        Yi_len = np.array([[1/a_i**2, 0], [0, 1/b_i **2]])
        Yj_len = np.array([[1/a_j ** 2, 0], [0, 1/b_j ** 2]])

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

        dGA = np.arccos(front_part * exp_part)
        return dGA
    except:
        # If we get get a crater projected with the semi major/minor axis = 0, this is an incorrect result therefore we max out the arccos.
        # Not that for GA, arcos maxes at pi/2 - but it should be okay to return pi.
        return math.pi

def ellipse_pair_distance_metric_gaussian_angle_ck(image_conics, craters_world, Pm_c, weighted_matrix, continuous=False, Pm_c_true=np.identity(1)):
    W = np.diag(weighted_matrix)
    avg_gaussian_angle = 0
    for c in range(len(image_conics)):
        Ai = image_conics[c]
        gauss_dist = gaussian_angle_ck(Ai, craters_world[c], Pm_c)**2
        avg_gaussian_angle += W[c]*gauss_dist #**2 removed square because acos should always be positive 
    return avg_gaussian_angle



def ellipse_centre_euclid_distance(A, crater, Pm_c, continuous=False, Pm_c_true=np.identity(1)):
    
    C = conic_from_crater(crater, Pm_c)

    x_a, y_a, _, _, _ = conic_matrix_to_ellipse(A)
    x_c, y_c, _, _, _ = conic_matrix_to_ellipse(C)
    euclid_dist = np.linalg.norm(np.array([x_a, y_a]) - np.array([x_c, y_c]))
    return euclid_dist

# Get the reprojection error of projected crater centres (image independent).
def ellipse_pair_distance_metric_ellipse_centre_euclid_distance(image_conics, craters_world, Pm_c, weighted_matrix, continuous=False, Pm_c_true=np.identity(1)):
    W = np.diag(weighted_matrix)
    err = 0
    for i, c in enumerate(craters_world):
        Ai = image_conics[i]
        err += W[i]*(ellipse_centre_euclid_distance(Ai, c, Pm_c))
    return err


def ellipse_distance(A, crater, Pm_c, continuous=False, Pm_c_true=np.identity(1)):
    
    C = conic_from_crater(crater, Pm_c)

    x_a, y_a, a_a, b_a, phi_a = conic_matrix_to_ellipse(A)
    x_c, y_c, a_c, b_c, phi_c = conic_matrix_to_ellipse(C)

    ellipse_distance = np.linalg.norm(np.array([x_a, y_a, a_a, b_a, phi_a]) - np.array([x_c, y_c, a_c, b_c, phi_c]))
    return ellipse_distance

def ellipse_pair_distance_metric_ellipse_distance(image_conics, craters_world, Pm_c, weighted_matrix, continuous=False, Pm_c_true=np.identity(1)):
    W = np.diag(weighted_matrix)
    avg_el_dist = 0
    s = 0
    for c in range(len(image_conics)):
        Ai = image_conics[c]
        el_dist = ellipse_distance(Ai, craters_world[c], Pm_c)
        avg_el_dist += W[c]*el_dist #removed square because acos should always be positive 
        s += el_dist

    return avg_el_dist

def characteristics_distance(A, crater, Pm_c, continuous=False, Pm_c_true=np.identity(1)):
    
    C = conic_from_crater(crater, Pm_c)

    points_a = extract_characteristic_points(A)
    points_c = extract_characteristic_points(C)

    ellipse_distance = np.linalg.norm(points_a - points_c)
    return ellipse_distance

def ellipse_characteristics_distance(image_conics, craters_world, Pm_c, weighted_matrix, continuous=False, Pm_c_true=np.identity(1)):
    W = np.diag(weighted_matrix)
    avg_el_dist = 0
    s = 0
    for c in range(len(image_conics)):
        Ai = image_conics[c]
        el_dist = characteristics_distance(Ai, craters_world[c], Pm_c)
        avg_el_dist += W[c]*el_dist #**2 removed square because acos should always be positive 
        s += el_dist

    return avg_el_dist

def wasserstein_distance(A, crater, Pm_c, continuous=False, Pm_c_true=np.identity(1)):
    
    C = conic_from_crater(crater, Pm_c)

    Ai = A
    Ci = C

    # Get ellipse parameters of image and projected conics.
    x_a, y_a, a_a, b_a, phi_a = conic_matrix_to_ellipse(Ai)
    mu_a = np.array([x_a, y_a])
    R_a = np.array([[math.cos(phi_a), -1*math.sin(phi_a)],[math.sin(phi_a), math.cos(phi_a)]])
    r_a = np.array([[1/(b_a**2), 0],[0, 1/(a_a**2)]])
    sig_inv_a = R_a @ r_a @ np.transpose(R_a)
    sig_a = (np.linalg.inv(sig_inv_a)) # absolute() - we get very small (e-12) negative numbers that need to be positive for sqrt calculations later on.

    x_c, y_c, a_c, b_c, phi_c = conic_matrix_to_ellipse(Ci)
    mu_c = np.array([x_c, y_c])
    R_c = np.array([[math.cos(phi_c), -1*math.sin(phi_c)],[math.sin(phi_c), math.cos(phi_c)]])
    r_c = np.array([[1/(b_c**2), 0],[0, 1/(a_c**2)]])
    sig_inv_c = R_c @ r_c @ np.transpose(R_c)
    sig_c = np.linalg.inv(sig_inv_c) # absolute() - we get very small (e-12) negative numbers that need to be positive for sqrt calculations later on.

    # TODO: do I need to square the norm?
    # TODO: do I need to sqare the result?
    err = (np.linalg.norm(mu_a - mu_c)**2 + np.trace(sig_a + sig_c - 2*scipy.linalg.sqrtm((scipy.linalg.sqrtm(sig_a) @ sig_c @ scipy.linalg.sqrtm(sig_a)))))
 
    return err

# Wasserstein Distance - Zins et. al.
def wasserstein_distance_metric(image_conics, craters_world, Pm_c, weighted_matrix, continuous=False, Pm_c_true=np.identity(1)):
    W = np.diag(weighted_matrix)
    wasserstein_distance_err = 0
    for i in range(len(image_conics)):
        # Get image conic and projected conic.
        Ai = image_conics[i]
        wasserstein_distance_err += W[i]*(wasserstein_distance(Ai, craters_world[i], Pm_c))**2
    
    return wasserstein_distance_err

# Generate sampling poins for level-set - Zins et. al.
def generate_sampling_points(C, count_az, count_dist, scale):
    points = []

    x_c, y_c, a, b, phi = conic_matrix_to_ellipse(C)
    points.append([x_c, y_c])

    Rot = np.array([[math.cos(phi), -1*math.sin(phi)],[math.sin(phi), math.cos(phi)]])
    d_dist = scale / count_dist
    d_az = 2 * math.pi / count_az

    for i in range(count_dist):
        for j in range(count_az):
            s = (i+1) * d_dist
            v = np.array([math.cos(d_az * j) * a * s, math.sin(d_az * j) * b * s])
            pt = np.array([x_c, y_c]) + Rot @ v
            points.append(pt)

    return points

# Get the metric error for each crater.
def get_metric_errors(metric_func, image_conics, craters_world, Pm_c):
    metric_errs = []
    for i in range(len(image_conics)):
        # Get image conic and projected conic.
        Ai = image_conics[i]
        err = metric_func(Ai, craters_world[i], Pm_c)
        metric_errs.append(err)
    return metric_errs


def level_set(A, crater, Pm_c, continuous=False, Pm_c_true=np.identity(1)):
    C = conic_from_crater(crater, Pm_c)

    # Get image conic and projected conic.
    Ai = A
    Ci = C

    # Used from CK's generate_oriented_bbox_points_cpu(...).
    num_sam = 10
    rotated_points_x = np.zeros(num_sam**2)
    rotated_points_y = np.zeros(num_sam**2)
    level_curve_a = np.zeros(num_sam**2)
    
    xc, yc, a, b, phi = conic_matrix_to_ellipse(Ai)

    x_samples = np.linspace(-a, a, num_sam)
    y_samples = np.linspace(-b, b, num_sam)

    x, y = np.meshgrid(x_samples, y_samples)

    R = np.array([[math.cos(phi), -math.sin(phi)],
                    [math.sin(phi), math.cos(phi)]])

    r = np.array([[1 / (a ** 2), 0], [0, 1 / (b ** 2)]])
    D_a = R @ r @ np.transpose(R)

    idx = 0
    for i in range(num_sam):
        for j in range(num_sam):
            point = np.array([x[i, j], y[i, j]])
            rotated_point = np.dot(R, point)
            rotated_points_x[idx] = rotated_point[0] + xc
            rotated_points_y[idx] = rotated_point[1] + yc

            disp_a = np.array([rotated_points_x[idx], rotated_points_y[idx]]) - np.array([xc, yc])
            level_curve_a[idx] = np.transpose(disp_a) @ D_a @ disp_a
            idx += 1

    
    x_c, y_c, a_c, b_c, phi_c = conic_matrix_to_ellipse(Ci)

    R_c = np.array([[math.cos(phi_c), -math.sin(phi_c)],
                    [math.sin(phi_c), math.cos(phi_c)]])

    R_c_T = np.transpose(R_c)

    r_c = np.array([[1.0 / (a_c ** 2), 0.0],
                    [0.0, 1.0 / (b_c ** 2)]])

    r_c_R_c_T = np.dot(r_c, R_c_T)

    D_c = np.dot(R_c, r_c_R_c_T)

    level_set_distance_ellipse = 0
    for idx in range(num_sam * num_sam):
        x_point = rotated_points_x[idx]
        y_point = rotated_points_y[idx]
        disp_c = np.array([x_point - x_c, y_point - y_c])
        D_c_disp_c = np.dot(D_c, disp_c)
        level_curve_c = np.dot(D_c_disp_c, disp_c)

        level_set_distance_ellipse += math.sqrt((level_curve_a[idx] - level_curve_c) ** 2)

    return level_set_distance_ellipse

def level_set_based_metric(image_conics, craters_world, Pm_c, weighted_matrix, continuous=False, Pm_c_true=np.identity(1)):
    W = np.diag(weighted_matrix)
    level_set_distance = 0
    for i in range(len(image_conics)):
        Ai = image_conics[i]
        level_set_error = level_set(Ai, craters_world[i], Pm_c)
        level_set_distance += W[i]*level_set_error**2 #TODO: should I square?
            
    return level_set_distance

# Get the indensity of each crater a function of the metric error.
def get_metric_intensities(metric_func, image_conics, craters_world, Pm_c, W):
    metric_intensities = get_metric_errors(metric_func, image_conics, craters_world, Pm_c)
    metric_intensities = [metric_intensity * np.diag(W)[i] for i, metric_intensity in enumerate(metric_intensities)]
    if max(metric_intensities) <= 0:
        return [0 for val in metric_intensities]
    return [val/max(metric_intensities) for val in metric_intensities]
        
# Introduce noisy crater matches
def noisy_crater_matches(craters_world, un_scaled_craters_world, proportion_incorrect_crater_matches = 0):
    switched_crater_indices = []
    # Introduce incorrect matches.
    if proportion_incorrect_crater_matches > 0:
        # Given two craters have to be incorrectly matched with each other, we return if we only have 3 corretly matched craters
        # i.e. we 4 craters or less avaliable.
        if (len(craters_world) <= 4):
            return craters_world, un_scaled_craters_world, switched_crater_indices
        
        if (len(craters_world) - (len(craters_world) * proportion_incorrect_crater_matches) < 3):
            num_incorrectly_matched_craters = int(len(craters_world) - 3)
        else:
            num_incorrectly_matched_craters = max(2,int(len(craters_world) * proportion_incorrect_crater_matches))

        all_indices = range(len(craters_world))
        crater_indices = random.sample(all_indices, num_incorrectly_matched_craters)
        for i in range(1, len(crater_indices)):
            switched_crater_indices.append(crater_indices[i])
        switched_crater_indices.append(crater_indices[0]) 
        store_switched_craters = []
        store_switched_un_scaled_craters = []
        # Switch world crater order so it doesn't match with the camera crater order.
        for i in range(num_incorrectly_matched_craters):
            store_switched_craters.append(craters_world[switched_crater_indices[i]])
            store_switched_un_scaled_craters.append(un_scaled_craters_world[switched_crater_indices[i]])
        for i in range(num_incorrectly_matched_craters):
            craters_world[crater_indices[i]] = store_switched_craters[i]
            un_scaled_craters_world[crater_indices[i]] = store_switched_un_scaled_craters[i]
    return craters_world, un_scaled_craters_world, switched_crater_indices

# Get an updated weighted matrix based on a predicted Pm_c
def m_estimators_weighted_matrix(metric_function, image_conics, craters_world, Pm_c, estimator="Tukey", c = 10):
    metric_errors = get_metric_errors(metric_function, image_conics, craters_world, Pm_c)
    weights = [1]*len(metric_errors)
    for i, err in enumerate(metric_errors):
        if estimator == "Tukey":
            if abs(err) <= c:
                weights[i] = ((1 - (err/c)**2)**2)
            else:
                weights[i] = (0)
        if estimator == "German McClure":
            weights[i] = 1/(1+err**2)**2
    return np.diag(weights)


from itertools import product

def minimize_3D_crater_distance(dual_craters, true_craters):
    # Number of nodes
    n = len(true_craters)

    # All possible combinations of vector selections (0 or 1 for each node)
    combinations = list(product([0, 1], repeat=n))
    
    # Initialize the minimum distance and best selection
    min_distance = float('inf')

    # Store euclidean distance and angular distance.
    all_euclid_distances = []
    all_angular_distances = []
    
    # Iterate through all combinations
    for comb in combinations:
        euclid_distances = []
        angular_distances = []
        selected_craters = []

        # Select vectors according to the current combination
        for j in range(n):
            selected_craters.append(dual_craters[j][comb[j]])

        # Calculate the total distance for the selected vectors
        for j in range(n):
            for k in range(j + 1, n):
                euclid_dist, ang_dist = f(selected_craters[j], selected_craters[k], true_craters[j], true_craters[k])
                euclid_distances.append(euclid_dist)
                angular_distances.append(ang_dist)

        all_euclid_distances.append(euclid_distances)
        all_angular_distances.append(angular_distances)

    all_euclid_distances = np.array(all_euclid_distances)
    all_angular_distances = np.array(all_angular_distances)

    max_angular_distance = np.pi
    min_angular_distance = 0
    max_euclidean_distance = np.max(all_euclid_distances)  # Max of all euclidean distances
    min_euclidean_distance = 0
    
    # Normalize Euclidean distances to the angular distance range
    for i in range(len(all_euclid_distances)):
        for j in range(len(all_euclid_distances[0])):
            all_euclid_distances[i][j] = ((all_euclid_distances[i,][j] - min_euclidean_distance) / 
                                          (max_euclidean_distance - min_euclidean_distance)) * \
                                          (max_angular_distance - min_angular_distance) + min_angular_distance

    # Weights
    weights = []
    for i in range(n):
        weights.append([0,0])
    for i, comb in enumerate(combinations):
        d = np.sum(all_euclid_distances[i, :]) + np.sum(all_angular_distances[i, :])
        if d < min_distance:
            min_distance = d
            for j in range(n):
                weights[j][comb[j]] = 1
                weights[j][(comb[j] + 1) % 2] = 0

    return weights, combinations

def f(crater1_est, crater2_est, crater1_true, crater2_true):
    # Example objective function: Euclidean distance and angular distance
    euclid_distance = abs(np.linalg.norm(crater1_est.get_crater_centre() - crater2_est.get_crater_centre()) - 
                          np.linalg.norm(crater1_true.get_crater_centre() - crater2_true.get_crater_centre()))

    angular_distance = abs(angle_3D_vectors(crater1_est.norm, crater2_est.norm) - 
                           angle_3D_vectors(crater1_true.norm, crater2_true.norm))

    return euclid_distance, angular_distance

def angle_3D_vectors(v1, v2):
    # Example function to calculate the angle between two 3D vectors
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angle






# Depending on the metric used, we filter the craters to be used by the pose estimation algorithm differently.
def filter_craters(craters_world, craters_cam, num_craters, image, metric, min_craters = 0, un_scaled_craters_world = []):
    # Check that we have detected enough craters.
    if len(craters_world) < min_craters:
        raise Exception("Crater filtering failed.")
    
    # If we have specified to use more craters than what is avaliable, use all craters avaliable.
    if num_craters > len(craters_world):
        num_craters = len(craters_world)
    
    # Only use top "num_craters" largest craters.
    # Sort craters_world, craters_cam, and un_scaled_craters_world according to crater axis length in craters_world.
    semi_major_axes = [c.a for c in craters_world]
    if (len(un_scaled_craters_world) > 0):
        zipped = list(zip(semi_major_axes, craters_world, craters_cam, un_scaled_craters_world))
        sorted_zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
        _,craters_world,craters_cam,un_scaled_craters_world = zip(*sorted_zipped)
    else:
        zipped = list(zip(semi_major_axes, craters_world, craters_cam))
        sorted_zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
        _,craters_world,craters_cam = zip(*sorted_zipped)

    # new_craters_world = []
    # new_craters_cam = []
    # new_un_scaled_craters_world = []
    # for i, crater in enumerate(un_scaled_craters_world):
    #     if crater.a > 6000:
    #         new_craters_world.append(craters_world[i])
    #         new_craters_cam.append(craters_cam[i])
    #         new_un_scaled_craters_world.append(un_scaled_craters_world[i])

    # craters_world = new_craters_world
    # new_craters_cam = craters_cam
    # new_un_scaled_craters_world = un_scaled_craters_world

    # Get top evenly spaced craters using Kmeans.
    cam_XY = np.array([[c.x,c.y] for c in craters_cam])
    kmeans = KMeans(n_clusters=min(len(craters_world), num_craters), random_state=0, n_init="auto").fit(cam_XY)
    unique_cluster = []
    new_craters_world = []
    new_craters_cam = []
    new_un_scaled_craters_world = []
    for i, k in enumerate(kmeans.labels_):
        if len(unique_cluster) == num_craters:
            break
        if k not in unique_cluster:
            unique_cluster.append(k)
            new_craters_world.append(craters_world[i])
            new_craters_cam.append(craters_cam[i])
            if (len(un_scaled_craters_world) > 0):
                new_un_scaled_craters_world.append(un_scaled_craters_world[i])
    craters_world = new_craters_world
    craters_cam = new_craters_cam
    un_scaled_craters_world = new_un_scaled_craters_world

    if (len(un_scaled_craters_world) > 0):
        return craters_world[:num_craters], un_scaled_craters_world[:num_craters], craters_cam[:num_craters], [0,1]
    
    return craters_world[:num_craters], un_scaled_craters_world, craters_cam[:num_craters], [0,1]


    # If we are using the metric of crater diameter, we want the craters used in the algorithm to be similar in size.
    filter_by_similar_craters = False
    if metric == "crater_diameter":
        filter_by_similar_craters = True

    # Remove all craters that are greater than the average degradational state.
    # crater_degradational_states = [crater_w.degradational_state for crater_w in craters_world]
    # avg_degradational_state = np.average(np.array(crater_degradational_states))
    # degradational_state_filter = [(crater_degradational_states >= avg_degradational_state*1.5)][0]
    # craters_world = [craters_world[i] for i in range(len(craters_world)) if degradational_state_filter[i]]
    # craters_cam = [craters_cam[i] for i in range(len(craters_cam)) if degradational_state_filter[i]]

    #Take top 1000 craters with largest diameter
    craters_world = craters_world[:1000]
    craters_cam = craters_cam[:1000]

    if (filter_by_similar_craters):
        # Only get top num craters of *same* diameter.
        # TODO: the float is scale dependant so change this.
        radius_diff_thresh =  5
        num_similar_craters = 100
        for i in range(len(craters_world)-num_similar_craters):
            if abs( craters_world[i].a - craters_world[i+num_similar_craters].a ) <= radius_diff_thresh:
                craters_world = craters_world[i:i+num_similar_craters]
                craters_cam = craters_cam[i:i+num_similar_craters]
                break
            if (i >= len(craters_world)-num_similar_craters-1):
                craters_world = craters_world[len(craters_world)-num_similar_craters:]
                craters_cam = craters_cam[len(craters_cam)-num_similar_craters:]
                break
        if len(craters_world) < num_craters:
            raise Exception("Crater filtering failed.")
    else:
        # Only keep non-overlapping craters
        new_craters_world = []
        new_craters_cam = []
        for i in range(len(craters_world)):
            ci = craters_world[i]
            intersect = False
            for j in range(len(new_craters_world)):
                if i != j:
                    cj = new_craters_world[j]
                    C1C2 = math.sqrt((ci.X - cj.X)**2 + (ci.Y - cj.Y)**2)
                    # If the crater doesn't intersect with any other crater
                    # If cj is inside ci, we can still can count ci - C1C2 <= ci.a - cj.a or
                    if ( C1C2 <= cj.a - ci.a or C1C2 < cj.a + ci.a):
                        intersect = True
                        break
            if not (intersect):
                new_craters_world.append(ci)
                new_craters_cam.append(craters_cam[i])
                
        craters_world = new_craters_world
        craters_cam = new_craters_cam
        if len(craters_world) < num_craters:
            raise Exception("Crater filtering failed.")

        # Only keep craters that are fully in the image and on the surface
        new_craters_world = []
        new_craters_cam = []
        for i in range(len(craters_world)):
            cw = craters_world[i]
            # if (cw.X + cw.a <= surface_width/2 and cw.X - cw.a >= -surface_width/2):
            #     if (cw.Y + cw.a <= surface_width/2 and cw.Y - cw.a >= -surface_width/2):
            image_width, image_height, _ = image.shape
            cc = craters_cam[i]
            if (cc.x + cc.a < image_width and cc.x - cc.a >= 0):
                if (cc.y + cc.a < image_height and cc.y - cc.a >= 0):
                    new_craters_world.append(cw)
                    new_craters_cam.append(craters_cam[i])
        craters_world = new_craters_world
        craters_cam = new_craters_cam
        if len(craters_world) < num_craters:
            raise Exception("Crater filtering failed.")

    # Get top evenly spaced craters using Kmeans.
    world_XY = np.array([[c.X,c.Y] for c in craters_world])
    kmeans = KMeans(n_clusters=min(len(craters_world), num_craters), random_state=0, n_init="auto").fit(world_XY)
    unique_cluster = []
    new_craters_world = []
    new_craters_cam = []
    for i, k in enumerate(kmeans.labels_):
        if len(unique_cluster) == num_craters:
            break
        if k not in unique_cluster:
            unique_cluster.append(k)
            new_craters_world.append(craters_world[i])
            new_craters_cam.append(craters_cam[i])
    craters_world = new_craters_world
    craters_cam = new_craters_cam
    if len(craters_world) < num_craters:
        raise Exception("Crater filtering failed.")

    # Return the average diameter of the list of craters
    crater_diameters = [crater_w.a*2 for crater_w in craters_world]
    avg_diameter = np.average(np.array(crater_diameters))

    return craters_world, craters_cam, avg_diameter, [crater_diameters[-1], crater_diameters[0]]
