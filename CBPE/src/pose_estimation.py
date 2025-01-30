import cv2
import copy
import math
import cmath 
from mpmath import mp
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from numpy import linspace
import os
from sklearn import linear_model, datasets
import scipy.linalg
from scipy import ndimage
import time

from src.camera_pose_visualisation import *
from src.get_data import *
from src.metrics import *
from src.extrinsics import *
from src.image_display import *
from src.pose_estimation import *

import warnings
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a ")
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")


# Get the position offset based on the type of metric used
def get_position_offset_error(rm, position_offset):
    position_offset_error = height_to_surface_metric(rm[2], position_offset)
    return position_offset_error

# Get the difference between the ray intersection with the sphere (moon) for the estimation and ground truth projection matrices.
def get_surface_observation_error(K, Pm_c, Pm_c_estimated, scale, min_offset):
    C = np.array([0,0,0]) # Centre of moon.
    r = 1737400

    ex = np.dot(np.linalg.inv(K),Pm_c)
    Tm_c = ex[0:3,0:3]
    U = Tm_c[2,:] # Principal axis.
    rm = np.dot(-1*np.linalg.inv(Tm_c),ex[:,3])
    P = np.array([v*scale + min_offset for v in rm])
    Q = P - C
    a = np.dot(U,U) # should be = 1
    b = 2*np.dot(U,Q)
    c = np.dot(Q,Q) - np.dot(r,r)
    d = np.dot(b,b) - 4*np.dot(a,c)  # discriminant of quadratic
    q = (b**2) - (4*a*c)  
    # find two solutions  
    t1 = (-b-cmath.sqrt(q))/(2*a)  
    t2 = (-b+cmath.sqrt(q))/(2*a)
    sphere_intersection_ground_truth = (P+t1*U).real

    ex_est = np.dot(np.linalg.inv(K),Pm_c_estimated)
    Tm_c_est = ex_est[0:3,0:3]
    U_est = Tm_c_est[2,:] # Principal axis.
    rm_est = np.dot(-1*np.linalg.inv(Tm_c_est),ex_est[:,3])
    P_est = np.array([v*scale + min_offset for v in rm_est])
    Q_est = P_est - C
    a_est = np.dot(U_est,U_est) # should be = 1
    b_est = 2*np.dot(U_est,Q_est)
    c_est = np.dot(Q_est,Q_est) - np.dot(r,r)
    d_est = np.dot(b_est,b_est) - 4*np.dot(a_est,c_est)  # discriminant of quadratic
    q_est = (b_est**2) - (4*a_est*c_est)  
    # find two solutions  
    t1_est = (-b_est-cmath.sqrt(q_est))/(2*a)  
    t2 = (-b_est+cmath.sqrt(q_est))/(2*a_est)
    # print(t1, t2)
    sphere_intersection_estimated = (P_est+t1_est*U_est).real

    return np.linalg.norm(sphere_intersection_ground_truth-sphere_intersection_estimated)

# Get the observed surface distance as a function of the pixel to ground resolution.
def get_surface_observation_error_as_function_of_pixel_resolution(craters_world, Pm_c, Pm_c_estimated, image_to_world_resolution):
    estimated_crater_projections = get_projected_crater_centres(craters_world, Pm_c_estimated)
    ground_truth_crater_projections = get_projected_crater_centres(craters_world, Pm_c)
    avg_projection_difference = 0
    for i in range(len(estimated_crater_projections)):
        diff = math.sqrt((estimated_crater_projections[i][0]-ground_truth_crater_projections[i][0])**2+(estimated_crater_projections[i][1]-ground_truth_crater_projections[i][1])**2)
        avg_projection_difference += diff
    avg_projection_difference /= len(estimated_crater_projections)
    return image_to_world_resolution*avg_projection_difference

# Solve pose through known attitude using least squares.
def solve_non_coplanar_conics_known_attitude(craters_world, image_conics, Pm_c, K, ransac = False):
    rejected_craters = np.zeros(len(craters_world))
    valid_crater_indicies = range(len(craters_world))
    for j in range(2):
        # Find scales si*Ci and build system of linear equations.
        # Solve A*x = b 
        # Where each point in A has size [2x3] and each point in b has size [2x1] and the number of points is the number of craters.
        A = np.asarray([[0]*3]*len(valid_crater_indicies)*2)
        A = np.asarray(A, dtype = 'float64')
        b = np.asarray([[0]* 1]*len(valid_crater_indicies)*2)
        b = np.asarray(b, dtype = 'float64')
        for i, ind in enumerate(valid_crater_indicies):
            k = np.array([0, 0, 1])
            S = np.vstack((np.eye(2), np.array([0,0])))
            Tl_m = craters_world[ind].get_ENU()#np.eye(3) # define a local coordinate system
            Pc_mi = craters_world[ind].get_crater_centre().reshape((3,1)) # get the real 3d crater point in moon coordinates
            Tm_l = np.transpose(Tl_m)
            extrinsics = np.dot(np.linalg.inv(K), Pm_c)
            Tm_c = extrinsics[0:3,0:3]
            Tc_m = np.linalg.inv(Tm_c)

            Ai = image_conics[ind]
            Ci = craters_world[ind].conic_matrix_local #TODO: I don't know if I'm allowed to scale the crater matrix.
            Bi = np.dot(Tc_m, np.dot(np.transpose(K),np.dot(Ai, np.dot(K, Tm_c))))

            # Find the scale si*Ci = Hci^-1*Ai*Hci
            Ci00 = np.matrix(np.transpose(np.dot(np.transpose(S),np.dot(Ci, S)))).getA1()
            sCi00 = np.matrix(np.dot(np.transpose(S),np.dot(Tm_l,np.dot(Bi,np.dot(Tl_m,S))))).getA1()
            si_hat = (np.dot(np.transpose(Ci00),sCi00))/np.dot(np.transpose(Ci00),Ci00)
            
            # Build system of linear equations.
            pointA = np.dot(np.transpose(S),np.dot(Tm_l, Bi))
            for r in range(len(pointA)):
                for c in range(len(pointA[r])):
                    A[i*2+r][c] = pointA[r][c]
            pointb = np.dot(np.transpose(S),np.dot(Tm_l, np.dot(Bi, Pc_mi))) - si_hat*np.dot(np.transpose(S),np.dot(Ci,k)).reshape((2,1))

            for r in range(len(pointb)):
                b[i*2+r] = pointb[r] 

        rm_hat = linear_model.LinearRegression().fit(A, b).coef_[0]
        if ransac:
            ransac = linear_model.RANSACRegressor(min_samples = 3)
            ransac.fit(A, b) #issue
            rm_hat = ransac.estimator_.coef_[0]
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)  
                    
        Pm_c_estimated = get_projection_matrix(Tm_c, rm_hat, K)
        res_euler = R.from_matrix(Tm_c).as_euler('zyx', degrees=True)

        if (j < 1):
            # Remove bad craters, or if too many removed, keep best three.
            errs = []
            ind = range(len(craters_world))
            valid_crater_indicies = []
            for i, c in enumerate(craters_world):
                err = gaussian_angle_ck(image_conics[i], c, Pm_c_estimated)
                errs.append(err)
                _, _, a_c, b_c, _ = conic_matrix_to_ellipse(Ai)
                sig_pix = 1
                sig = sig_pix*(0.85/math.sqrt(a_c*b_c))
                if err**2/sig**2 > 13.276:
                    rejected_craters[i] = 1
                else:
                    valid_crater_indicies.append(i)
            num_samples = 5 # We need 3 points to compute pose and 7 points to select from RANSAC
            if (len(craters_world) - np.array(rejected_craters).sum() < num_samples):
                sorted_ind = [x for _,x in sorted(zip(errs,ind))]
                rejected_craters = np.ones(len(craters_world))
                for i in sorted_ind[:num_samples]:
                    rejected_craters[i] = 0
                valid_crater_indicies = sorted_ind[:num_samples]

    rm_hat = linear_model.LinearRegression().fit(A, b).coef_[0]

    if ransac:
        ransac = linear_model.RANSACRegressor(min_samples = 3)
        ransac.fit(A, b)
        rm_hat = ransac.estimator_.coef_[0]   
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
               

    return Pm_c_estimated, res_euler, rm_hat

def projection_matrix_from_euler_and_position(K, euler_angles, position):
    if len(euler_angles) == 1 and len(euler_angles[0]) == 3:
        euler_angles = euler_angles[0]
    # Constrain the euler bounds to [-180, 180], [-90, 90], [-180, 180] 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html 
    euler_angles[0] = get_valid_euler_angle(-180, 180, euler_angles[0])
    euler_angles[1] = get_valid_euler_angle(-90, 90, euler_angles[1])
    euler_angles[2] = get_valid_euler_angle(-180, 180, euler_angles[2])
    Tm_c = R.from_euler('zyx', euler_angles, degrees=True).as_matrix()
    rm = np.array(position).reshape((3,1))
    Pm_c = K @ Tm_c @ (np.hstack((np.eye(3),-1*rm)))
    return Pm_c

# Returns a valid euler angle provided bound limit.
def get_valid_euler_angle(min_euler_angle, max_euler_angle, angle):
    if angle >= min_euler_angle and angle <= max_euler_angle:
        return angle
    else:
        if angle < min_euler_angle:
            diff = abs(angle - min_euler_angle)
            return max_euler_angle - diff
        diff = abs(angle - max_euler_angle)
        return min_euler_angle + diff
    
# Scipy's minimise.
def func_optimise_position(x, obj_func, image_conics, craters_world, K, euler_angles, W, continuous=False, Pm_c_true=np.identity(1)):
    return obj_func(image_conics, craters_world, projection_matrix_from_euler_and_position(K, euler_angles, x), W, continuous, Pm_c_true)

# Scipy's minimise.
def func(x, obj_func, image_conics, craters_world, K, W, continuous=False, Pm_c_true=np.identity(1)):
    st = "x0 = np.array(["
    for i in x:
        st+= str(i) +", "
    st = st[:-2]+"])"
    return obj_func(image_conics, craters_world, projection_matrix_from_euler_and_position(K, x[:3], x[3:]), W, continuous, Pm_c_true)

# Scipy's minimise.
def func_maass(x, obj_func, cubic_dual_craters, craters_world, K, W, continuous=False, Pm_c_true=np.identity(1)):
    st = "x0 = np.array(["
    for i in x:
        st+= str(i) +", "
    st = st[:-2]+"])"
    return obj_func(cubic_dual_craters, craters_world, projection_matrix_from_euler_and_position(K, x[:3], x[3:]), K, W, continuous, Pm_c_true)


# Scipy's minimise using dual craters but some weights are provided.
def func_dual_optimise_weights(x, dual_obj_func, dual_craters, Tm_c_euler, rm, craters_world, K, dual_method, W, continuous=False, Pm_c_true=np.identity(1)):
    st = "x0 = np.array(["
    for i in x:
        st+= str(i) +", "
    st = st[:-2]+"])"
    return dual_obj_func(dual_craters, x, craters_world, projection_matrix_from_euler_and_position(K, Tm_c_euler, rm), K, dual_method, W, continuous, Pm_c_true)


# Scipy's minimise using dual craters but some weights are provided.
def func_dual_some_weights(x, dual_obj_func, dual_craters, dual_weights, craters_world, K, dual_method, W, continuous=False, Pm_c_true=np.identity(1)):
    st = "x0 = np.array(["
    for i in x:
        st+= str(i) +", "
    st = st[:-2]+"])"
    dual_crater_weights = dual_weights
    return dual_obj_func(dual_craters, dual_crater_weights, craters_world, projection_matrix_from_euler_and_position(K, x[:3], x[3:6]), K, dual_method, W, continuous, Pm_c_true)

# # Scipy's minimise using dual craters but weights are provided.
# def func_dual_nested_inner_weight(x, dual_obj_func, dual_craters, dual_weights_init, craters_world, K, dual_method, W, continuous=False, Pm_c_true=np.identity(1)):
#     st = "x0 = np.array(["
#     for i in x:
#         st+= str(i) +", "
#     st = st[:-2]+"])"
#     weight_results = scipy.optimize.minimize(func_dual_no_pose, dual_weights_init, args=(dual_obj_func, dual_craters, x[:3], x[3:6], craters_world, K, dual_method, W, continuous, Pm_c_true), method="Powell", jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options={"disp":False})
    
#     return res = dual_obj_func(dual_craters, weight_results.x, craters_world, projection_matrix_from_euler_and_position(K, x[:3], x[3:6]), K, dual_method, W, continuous, Pm_c_true)



# Scipy's minimise using dual craters but weights are provided.
def func_dual_optimise_pose(x, dual_obj_func, dual_craters, dual_weights, craters_world, K, dual_method, W, continuous=False, Pm_c_true=np.identity(1)):
    st = "x0 = np.array(["
    for i in x:
        st+= str(i) +", "
    st = st[:-2]+"])"
    dual_crater_weights = dual_weights
    return dual_obj_func(dual_craters, dual_crater_weights, craters_world, projection_matrix_from_euler_and_position(K, x[:3], x[3:6]), K, dual_method, W, continuous, Pm_c_true)


# Scipy's minimise using dual craters.
def func_dual(x, dual_obj_func, dual_craters, craters_world, K, dual_method, W, continuous=False, Pm_c_true=np.identity(1)):
    st = "x0 = np.array(["
    for i in x:
        st+= str(i) +", "
    st = st[:-2]+"])"
    dual_crater_weights = x[6:]
    return dual_obj_func(dual_craters, dual_crater_weights, craters_world, projection_matrix_from_euler_and_position(K, x[:3], x[3:6]), K, dual_method, W, continuous, Pm_c_true)

def func_optimise_weights_mass(x0_weights, dual_craters, K):
    # Scale all points.
    # Scale everything by the largest distance from a crater to the centre of the moon.
    crater_distances = []
    for i in range(len(x0_weights)):
        for j in range(2):
            crater_dual = dual_craters[i][j]
            crater_distances.append(np.linalg.norm([crater_dual.X, crater_dual.Y, crater_dual.Z]))
    scale = max(crater_distances)
    offset = 0
    scaled_dual_craters = []
    for i in range(len(x0_weights)):
        crater_dual = dual_craters[i] 
        scaled_dual_craters.append((Crater_dual_c((crater_dual[0].X - offset)/scale, (crater_dual[0].Y - offset)/scale, (crater_dual[0].Z - offset)/scale, (crater_dual[0].r)/scale, crater_dual[0].norm), Crater_dual_c((crater_dual[1].X - offset)/scale, (crater_dual[1].Y - offset)/scale, (crater_dual[1].Z - offset)/scale, (crater_dual[1].r)/scale, crater_dual[1].norm)))    

    surface_norms = []
    surface_points = []
    image_points = []
    for i, weight in enumerate(x0_weights):
        weight = int(weight)
        surface_norms.append(scaled_dual_craters[i][weight].norm)
        surface_points.append(scaled_dual_craters[i][weight].get_crater_centre())
        projected_crater_centre = scaled_dual_craters[i][weight].proj_crater_centre(np.hstack((K,np.array([0,0,0]).reshape((3,1)))))
        image_points.append(projected_crater_centre)

    energy = cubic_spline_energy(image_points,surface_norms,surface_points)

    return energy

# Calls the respective pose estimation method.
def pose_estimation(pose_method, craters_world, un_scaled_craters_world, craters_cam, num_craters, min_craters_available, Pm_c, un_normalised_Pm_c, K, distCoeffs, image, noise_pixel_offset, max_noise_sigma_pix, dir, write_dir, noise_directory, file_name, add_noise, extrinsics_visualisation, craters_world_dir, crater_world_filenames, metric, euler_bound, position_bound, un_normalised_position_bound, proportion_incorrect_crater_matches, min_offset, scale, is_pangu, m_estimator = None, tuning_const = 1, factr_ = 10000000.0, pgtol_=1e-05, epsilon_=1e-08, maxfun_=15000, maxiter_=15000, maxls_=20, seed=None, continuous=False, propagated_position=None):
    start_time = time.perf_counter()
    # if euler_bound == 0:
    #     euler_bound = 1e-20
    if position_bound == 0:
        position_bound = 1e-20

    # # Store all the craters in the map separately for extrinsic plotting later.
    # all_craters_world = copy.deepcopy(craters_world)
    
    # Store crater diameters (m).
    crater_diameters_m = [c.a for c in un_scaled_craters_world]

    # Filter the craters.
    try:
        craters_world, un_scaled_craters_world, craters_cam, crater_diameter_range = filter_craters(craters_world, craters_cam, num_craters, image, metric, min_craters = min_craters_available, un_scaled_craters_world=un_scaled_craters_world)
    except:
        # print("ERROR - NOT ENOUGH CRATERS DETECTED")
        raise Exception("Crater filtering failed.")

    # Get average and std projected crater size.
    # Get distribution of the craters.
    crater_sizes = []
    grid_size = 4
    h, w, _ = image.shape
    crater_appearences = np.zeros((grid_size,grid_size))
    col_step_size = int(w/grid_size)
    row_step_size = int(h/grid_size)
    crater_x_cs = []
    crater_y_cs = []

    for c in un_scaled_craters_world:
        x_c, y_c, a, b, _ = conic_matrix_to_ellipse(conic_from_crater(c, un_normalised_Pm_c))
        crater_x_cs.append(x_c)
        crater_y_cs.append(y_c)
        crater_sizes.append(a*b*math.pi)
        for col in range(0, w, col_step_size):
            for row in range(0, h, row_step_size):
                if x_c >= col and x_c <= col + col_step_size and y_c >= row and y_c <= row + row_step_size:
                    crater_appearences[int(col/(col_step_size)), int(row/(row_step_size))] += 1
    crater_appearence_std = np.std(crater_appearences.flatten())

    # Get the curvature energy of the surface.
    if scale != 1:
        true_energy = surface_energy(craters_world, Pm_c)
    else:
        crater_distances = []
        for crater in craters_world:
            crater_distances.append(np.linalg.norm([crater.X, crater.Y, crater.Z]))
        scale_test = max(crater_distances)
        
        offset = 0
        scaled_craters_world = []
        for i, crater in enumerate(craters_world):
            if not is_pangu:
                scaled_craters_world.append(Crater_w_scaled((crater.X - offset)/scale_test, (crater.Y - offset)/scale_test, (crater.Z - offset)/scale_test, (crater.a)/scale_test, (crater.b)/scale_test, crater.phi,crater.id,is_pangu, crater.norm))
            else:
                scaled_craters_world.append(Crater_w_scaled((crater.X - offset)/scale_test, (crater.Y - offset)/scale_test, (crater.Z - offset)/scale_test, (crater.a)/scale_test, (crater.b)/scale_test, crater.phi,crater.id,is_pangu=is_pangu))
        
        k_extrinsic_matrix_ground_truth = np.dot(np.linalg.inv(K),Pm_c)
        R_gt = k_extrinsic_matrix_ground_truth[0:3,0:3]
        ground_truth_world_position = -1*np.dot(np.linalg.inv(R_gt), k_extrinsic_matrix_ground_truth[:,3])
        ground_truth_world_position = np.array([v/scale_test + min_offset for v in ground_truth_world_position])

        ground_truth_world_position = np.dot((R_gt), -1*ground_truth_world_position).reshape((3,1))
        k_extrinsic_matrix_ground_truth = np.dot((K),np.hstack((R_gt,ground_truth_world_position)))

        true_energy = surface_energy(scaled_craters_world, k_extrinsic_matrix_ground_truth)


    # Get line of best fit through the crater centre points.
    m_slope, c_offset = np.polyfit(crater_x_cs, crater_y_cs, 1)
    p1 = np.array([0, int(c_offset)])
    p2 = np.array([w, int(m_slope*w+c_offset)])

    # Get perpendicular crater distances from line of best fit.
    distances_from_line_of_best_fit = []
    for j in range(len(crater_x_cs)):
        p3 = np.array([crater_x_cs[j],crater_y_cs[j]])
        d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
        distances_from_line_of_best_fit.append(d)

    # Get the corresponding 3D and 2D crater centre points. 
    # TODO: use parsed crater_cam craters instead (currently not working in optimisation even though projecting correctly on the image frame).
    # Rather than use the projected image conics provided in a separate file, we project the world crater conics and use these instead.
    # image_conics = np.array([conic_from_crater(c, Pm_c, add_noise, noise_pixel_offset, max_noise_sigma_pix) for c in craters_world])
    image_conics = []
    un_normalised_image_conics = []
    # Keep crater perturbation noise constant across every pose esimation.

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    for i in range(len(craters_world)):
        image_conic, un_normalised_image_conic = normalised_and_un_normalised_conic_from_crater(craters_world[i], un_scaled_craters_world[i], craters_cam[i],Pm_c, un_normalised_Pm_c, add_noise, noise_pixel_offset, max_noise_sigma_pix, continuous)
        image_conics.append(image_conic)
        un_normalised_image_conics.append(un_normalised_image_conic)
    image_conics = np.array(image_conics)
    un_normalised_image_conics = np.array(un_normalised_image_conics)


    # Introduce noisy crater matches
    craters_world, un_scaled_craters_world, switched_crater_indices = noisy_crater_matches(craters_world, un_scaled_craters_world, proportion_incorrect_crater_matches)
    correctly_matched_crater_indices = []
    for i in range(len(craters_world)):
        if i not in switched_crater_indices:
            correctly_matched_crater_indices.append(i)

    # Normalised intrinsics/extrinsics.
    extrinsics = np.dot(np.linalg.inv(K), Pm_c)
    Tm_c = extrinsics[0:3,0:3]
    Tm_c_euler = R.from_matrix(Tm_c).as_euler('zyx', degrees=True)
    rc = extrinsics[0:3, 3]
    rm = -1*np.dot(np.linalg.inv(Tm_c), rc)

    # Un-normalised intrinisics/extrinsics.
    un_normalised_extrinsics = np.dot(np.linalg.inv(K), un_normalised_Pm_c)
    un_normalised_Tm_c = un_normalised_extrinsics[0:3,0:3]
    un_normalised_Tm_c_euler = R.from_matrix(un_normalised_Tm_c).as_euler('zyx', degrees=True)
    un_normalised_rc = un_normalised_extrinsics[0:3, 3]
    un_normalised_rm = -1*np.dot(np.linalg.inv(un_normalised_Tm_c), un_normalised_rc)
        
    metric_function = ellipse_pair_distance_metric_ellipse_centre_euclid_distance
    intensity_func = ellipse_centre_euclid_distance
    if pose_method == "6DoF_level_set_based_bounded":
        metric_function = level_set_based_metric
        intensity_func = level_set
        tuning_const = 1000
    elif pose_method == "6DoF_wasserstein_distance_bounded":
        metric_function = wasserstein_distance_metric
        intensity_func = wasserstein_distance
        tuning_const = 20000
    elif pose_method == "6DoF_gaussian_angle_bounded":
        metric_function = ellipse_pair_distance_metric_gaussian_angle_ck
        intensity_func = gaussian_angle_ck
        tuning_const = 0.7
    elif pose_method == "PnC":
        metric_function = ellipse_pair_distance_metric_ellipse_distance
        intensity_func = ellipse_distance
        tuning_const = 100 #100
    elif pose_method == "6DoF_ellipse_characteristics_bounded":
        metric_function = ellipse_characteristics_distance
        intensity_func = characteristics_distance
        tuning_const = 500 #TODO:  check this tuning constant
    elif pose_method == "6DoF_euclidean_distance_bounded":
        metric_function = ellipse_pair_distance_metric_ellipse_centre_euclid_distance
        intensity_func = ellipse_centre_euclid_distance
        tuning_const = 40 #90
    elif pose_method == "PnP":
        metric_function = crater_centre_pnp_metric
        intensity_func = crater_centre_pnp
        tuning_const = 40
    elif pose_method == "3Dof_least_squares":
        metric_function = ellipse_pair_distance_metric_gaussian_angle_ck
        intensity_func = gaussian_angle_ck
    elif pose_method == "3Dof_least_squares_RANSAC":
        metric_function = ellipse_pair_distance_metric_gaussian_angle_ck
        intensity_func = gaussian_angle_ck
    elif pose_method == "dual_PnP_linear":
        metric_function = dual_pnp_reprojection_error
        intensity_func = dual_reprojection_error
        dual_method = "linear"
        tuning_const = 40
    elif pose_method == "dual_PnP_sigmoid":
        metric_function = dual_pnp_reprojection_error
        intensity_func = dual_reprojection_error
        dual_method = "sigmoid"
        tuning_const = 40
    elif pose_method == "dual_PnP_sigmoid_permutations":
        metric_function = dual_pnp_reprojection_error
        intensity_func = dual_reprojection_error
        dual_method = "sigmoid"
        tuning_const = 40
    elif pose_method == "maass_cubic_pnp":
        metric_function = maass_pnp_reprojection_error
        intensity_func = maass_reprojection_error
        dual_method = "linear"
        tuning_const = 40
    elif pose_method == "maass_cubic_pnp_optimised":
        metric_function = dual_pnp_reprojection_error
        intensity_func = dual_reprojection_error
        dual_method = "sigmoid"
        tuning_const = 40



    # Store all results to get the smallest.
    errs = []
    res_eulers = []
    rm_ests = []
    opt_res = []
    dual_weights = []

    # TODO: remove
    inverse_true = False

    # Set weighted matrix for m-estimators.
    W = np.identity(len(craters_world)) # weighted matrix.
    # random.seed(None)
    fp = 0 # An incorrectly matched crater that was identified as a correct match.
    fn = 0 # A correctly matched crater that was identified as an incorrect match.

    # Set bounds.
    rm_bounds = [(rm[0]-position_bound, rm[0]+position_bound),(rm[1]-position_bound, rm[1]+position_bound),(rm[2]-position_bound, rm[2]+position_bound)]
    euler_bounds = [(Tm_c_euler[0]-euler_bound, Tm_c_euler[0]+euler_bound),(Tm_c_euler[1]-euler_bound, Tm_c_euler[1]+euler_bound),(Tm_c_euler[2]-euler_bound, Tm_c_euler[2]+euler_bound)]
    bounds_x0 = [euler_bounds[0], euler_bounds[1], euler_bounds[2], rm_bounds[0], rm_bounds[1], rm_bounds[2]]

    # Set initialisers.
    # x0_euler = [random.uniform(euler_bounds[0][0], euler_bounds[0][1]), random.uniform(euler_bounds[1][0], euler_bounds[1][1]), random.uniform(euler_bounds[2][0], euler_bounds[2][1])] # Given the euler bounds are so small, randomly initialise it once.
    # x0s = []
    # num_position_points_1D = 2
    # for X in np.linspace(rm_bounds[0][0], rm_bounds[0][1], num_position_points_1D):
    #     for Y in np.linspace(rm_bounds[1][0], rm_bounds[1][1], num_position_points_1D):
    #         for Z in np.linspace(rm_bounds[2][0], rm_bounds[2][1], num_position_points_1D):
    #             x0s.append([x0_euler[0], x0_euler[1], x0_euler[2], X, Y, Z])
    #             # for yaw in np.linspace(euler_bounds[0][0], euler_bounds[0][1], num_position_points_1D):
    #             #     for pitch in np.linspace(euler_bounds[1][0], euler_bounds[1][1], num_position_points_1D):
    #             #         for roll in np.linspace(euler_bounds[2][0], euler_bounds[2][1], num_position_points_1D):
    #             #             x0s.append([yaw, pitch, roll, X, Y, Z])
    # # Uncomment if you want random initialisation.
    x0s = [[random.uniform(euler_bounds[0][0], euler_bounds[0][1]), random.uniform(euler_bounds[1][0], euler_bounds[1][1]), random.uniform(euler_bounds[2][0], euler_bounds[2][1]), random.uniform(rm_bounds[0][0], rm_bounds[0][1]), random.uniform(rm_bounds[1][0], rm_bounds[1][1]), random.uniform(rm_bounds[2][0], rm_bounds[2][1])]]



    # NOTE: IF YOU WANT TO COMPUTE COMBINATIONS
    show_all_combinations = False
    for j, x0 in enumerate(x0s):
        bounds_x0 = [(x0[0]-euler_bound, x0[0]+euler_bound), (x0[1]-euler_bound, x0[1]+euler_bound), (x0[2]-euler_bound, x0[2]+euler_bound), (x0[3]-position_bound, x0[3]+position_bound), (x0[4]-position_bound, x0[4]+position_bound), (x0[5]-position_bound, x0[5]+position_bound)]
        if propagated_position:
            # Only apply the propagated position IF it is better that the initialised estimate (i.e. within the expected bounds of the initialised estimate) 
            if all(bounds_x0[i][0] <= propagated_position[i-3] <= bounds_x0[i][1] for i in range(3,6)):
                x0[3] = propagated_position[0]
                x0[4] = propagated_position[1]
                x0[5] = propagated_position[2]

        # Return early if the special case of known attitude for Christian's non-coplanar conics method.
        if pose_method == "3Dof_least_squares" or pose_method == "3Dof_least_squares_RANSAC":
            ransac = False
            if pose_method == "3Dof_least_squares_RANSAC":
                ransac = True
            Pm_c_init = projection_matrix_from_euler_and_position(K, Tm_c_euler, x0[3:])
            
            Pm_c_estimated, res_euler, rm_est = solve_non_coplanar_conics_known_attitude(craters_world, image_conics, Pm_c_init, K, ransac)
            err = metric_function(image_conics, craters_world, Pm_c_estimated, W)

            errs.append(err)
            res_eulers.append(res_euler)
            rm_ests.append(rm_est)
            opt_res.append(None)

        elif pose_method == "maass_cubic_pnp":
            dual_craters = dual_crater_from_imaged_ellipse(image_conics, craters_world, K, Pm_c)
            true_duals, true_duals_indices = find_crater_norms_from_cubic_spline_interpolation(craters_world, dual_craters, K, is_pangu, scale_data=False)


            results = scipy.optimize.fmin_l_bfgs_b(func_maass, x0, fprime=None, args=(metric_function, true_duals, craters_world, K, W, continuous, Pm_c), approx_grad=1, bounds=bounds_x0, m=10, factr=factr_, pgtol=pgtol_, epsilon=epsilon_, iprint=-1, maxfun=maxfun_, maxiter=maxiter_, disp=False, callback=None, maxls=maxls_)

            errs.append(results[1])
            res_eulers.append([results[0][0], results[0][1], results[0][2]])
            rm_ests.append([results[0][3], results[0][4], results[0][5]])
            opt_res.append(results[0])    

            # Inverse the indices to get weights.
            weights_est = []
            scales_est = []
            for i in range(len(true_duals_indices)):
                if true_duals_indices[i] == 0:
                    weights_est.append(1)
                    scales_est.append(30)
                else:
                    weights_est.append(0)
                    scales_est.append(-30)      

        elif pose_method == "maass_cubic_pnp_optimised":
            dual_craters = dual_crater_from_imaged_ellipse(image_conics, craters_world, K, Pm_c)

            # Greedy, non optimised approach.
            _, true_duals_indices = find_crater_norms_from_cubic_spline_interpolation(craters_world, dual_craters, K, is_pangu, scale_data=True)
            true_duals_indices = np.around(true_duals_indices)
       

            # Inverse the indices to get weights.
            prev_weights_est = []
            for i in range(len(true_duals_indices)):
                if true_duals_indices[i] == 0:
                    prev_weights_est.append(1)
                else:
                    prev_weights_est.append(0)

            for w in range(len(craters_world)):
                maximum_possible_scale = 30
                max_scale = 0.1 #1.2 for noise and 0.1 without noise
                min_scale = -1*max_scale
                if (prev_weights_est[w] == 0):
                    x0.append(min_scale)
                else:
                    x0.append(max_scale)
                bounds_x0.append((min_scale, max_scale))

            current_max_scale = max_scale 
            current_min_scale = min_scale
            
            # Optimise weights and pose simultaneously using the sigmoid function.
            it = 0
            max_it = 15
            fval_prev = math.inf
            tol = 1e-10

            while True:
                it+=1

                results = scipy.optimize.minimize(func_dual, x0, args=(metric_function, dual_craters, craters_world, K, dual_method, W, continuous, Pm_c), method="SLSQP", jac=None, hess=None, hessp=None, bounds=bounds_x0, constraints=(), tol=None, callback=None, options={"disp":False, "ftol":1e-10})
                # results = scipy.optimize.fmin_l_bfgs_b(func_dual, x0, fprime=None, args=(metric_function, dual_craters, craters_world, K, dual_method, W, continuous, Pm_c), approx_grad=1, bounds=bounds_x0, m=10, factr=factr_, pgtol=pgtol_, epsilon=epsilon_, iprint=-1, maxfun=maxfun_, maxiter=maxiter_, disp=False, callback=None, maxls=maxls_)

                
                fval = results.fun #results[1]
                # print([current_min_scale,current_max_scale],results.x[6:])
                # TODO: (Sofia) Terminate when rounded weights don't change?
                if abs(fval - fval_prev) < tol or it >= max_it or current_max_scale >= maximum_possible_scale:
                    # print("termination:",abs(fval - fval_prev) < tol,it >= max_it,current_max_scale >= maximum_possible_scale)
                    
                    # In the case of planar surfaces, optimiser might reach a local min using all the wrong duals.
                    # We check to see if the polar flip of the scales is produces a better fval, and if so, we replace the results.
                    polar_res = results.x[:6]
                    for i in (-1*results.x[6:]):
                        polar_res = np.append(polar_res,i)
                    results_polar = scipy.optimize.minimize(func_dual, polar_res, args=(metric_function, dual_craters, craters_world, K, dual_method, W, continuous, Pm_c), method="SLSQP", jac=None, hess=None, hessp=None, bounds=bounds_x0, constraints=(), tol=None, callback=None, options={"disp":False, "ftol":1e-10})
                    
                    if results_polar.fun < fval:
                        results = results_polar
                        inverse_true = True
                
                    scales_est =results.x[6:]#results[0][6:]
                    weights_est = [get_weight(dual_method,s) for s in scales_est]
                    break

                # Adjust search bounds
                # TODO: (Sofia) I don't know if this makes sense - fval should always be lower with higher scale (?).
                current_min_scale = current_min_scale + current_min_scale/2
                current_max_scale = current_max_scale + current_max_scale/2
    
                fval_prev = fval
                x0[:6] = results.x[:6]#results[0][:6]

                x0[6:] = results.x[6:]#results[0][6:]
                for i in range(len(craters_world)):
                    if round(x0[6+i]) < 0:
                        x0[6+i] = current_min_scale
                    else:
                        x0[6+i] = current_max_scale
                    # x0[6:] = [-current_max_scale, current_max_scale, -current_max_scale, current_max_scale]
                    bounds_x0[6+i] = (current_min_scale, current_max_scale) 

            error = results.fun
            errs.append(error)
            res_eulers.append(results.x[:3])
            rm_ests.append(results.x[3:6])
            opt_res.append(results.x) 

            pose_est = results.x[:6] #results[0][:6]
            dual_weights_est = np.around(weights_est)
            dual_weights.append(dual_weights_est)

            print("",prev_weights_est,"\n",dual_weights_est)

        elif pose_method == "dual_PnP_linear" or pose_method == "dual_PnP_sigmoid" or pose_method == "dual_PnP_sigmoid_permutations":
            # # TODO:(Sofia) add m-estimator code here too.

            # # Get dual circles and set up optimisation for weights.
            dual_craters = dual_crater_from_imaged_ellipse(image_conics, craters_world, K, Pm_c)

            for w in range(len(craters_world)):
                if dual_method == "sigmoid":
                    max_scale = 0.1 #1.2 for noise and 0.1 without noise
                    min_scale = -1*max_scale
                    init_dual_scale = random.uniform(min_scale, max_scale)
                    bounds_x0.append((min_scale, max_scale))
                else:
                    max_scale = 1
                    min_scale = 0
                    init_dual_scale = random.randint(min_scale, max_scale)
                    bounds_x0.append((min_scale, max_scale))
                
                x0.append(init_dual_scale)
                maximum_possible_scale = 30

            if pose_method == "dual_PnP_sigmoid_permutations":
                combinations = list(product([0, 1], repeat=len(craters_world)))

                ## Try all scale combinations
                scaled_combinations = []
                for i, combination in enumerate(combinations):
                    scales = []
                    for j in range(len(combination)):
                        if combination[j] == 1:
                            scales.append(maximum_possible_scale)
                        else:
                            scales.append(-1*maximum_possible_scale)
                    scaled_combinations.append(scales)

                all_results = []
                fvals = []
                true_res = []

                x0_true = [Tm_c_euler[0],Tm_c_euler[1],Tm_c_euler[2],rm[0],rm[1],rm[2]]
                for scaled_combination in scaled_combinations:
                    true_res.append((func_dual_optimise_pose(x0_true, metric_function, dual_craters, scaled_combination, craters_world, K, dual_method, W, continuous, Pm_c)))
                true_res_index = np.argmin(np.array(true_res))
                true_scaled_combination = scaled_combinations[true_res_index]

                for scaled_combination in scaled_combinations:
                    current_results = scipy.optimize.minimize(func_dual_optimise_pose, x0[:6], args=(metric_function, dual_craters, scaled_combination, craters_world, K, dual_method, W, continuous, Pm_c), method="SLSQP", jac=None, hess=None, hessp=None, bounds=bounds_x0[:6], constraints=(), tol=None, callback=None, options={"maxiter":50,"disp":False, "ftol":1e-10,"eps":1e-8})
                    all_results.append(current_results)
                true_res_index = np.argmin(np.array(true_res))
                all_fvals = np.array([result.fun for result in all_results])
                results = all_results[np.argmin(all_fvals)]
                scales_est = scaled_combinations[np.argmin(all_fvals)]
                weights_est = [get_weight(dual_method,s) for s in scales_est]

            elif dual_method == "sigmoid":
                # TODO:(Sofia) add m-estimator code here too.
                sigmoid_step = 0.1 #0.001 #sigmoid_limit/1000
                current_max_scale = max_scale 
                current_min_scale = min_scale
                
                # Optimise weights and pose simultaneously using the sigmoid function.
                it = 0
                max_it = 15
                fval_prev = math.inf
                tol = 1e-10
                

                while True:
                    it+=1

                    results = scipy.optimize.minimize(func_dual, x0, args=(metric_function, dual_craters, craters_world, K, dual_method, W, continuous, Pm_c), method="SLSQP", jac=None, hess=None, hessp=None, bounds=bounds_x0, constraints=(), tol=None, callback=None, options={"disp":False, "ftol":1e-10})
                    # results = scipy.optimize.fmin_l_bfgs_b(func_dual, x0, fprime=None, args=(metric_function, dual_craters, craters_world, K, dual_method, W, continuous, Pm_c), approx_grad=1, bounds=bounds_x0, m=10, factr=factr_, pgtol=pgtol_, epsilon=epsilon_, iprint=-1, maxfun=maxfun_, maxiter=maxiter_, disp=False, callback=None, maxls=maxls_)

                    
                    fval = results.fun #results[1]
                    # print([current_min_scale,current_max_scale],results.x[6:])
                    # TODO: (Sofia) Terminate when rounded weights don't change?
                    if abs(fval - fval_prev) < tol or it >= max_it or current_max_scale >= maximum_possible_scale:
                        
                        # In the case of planar surfaces, optimiser might reach a local min using all the wrong duals.
                        # We check to see if the polar flip of the scales is produces a better fval, and if so, we replace the results.
                        polar_res = results.x[:6]
                        for i in (-1*results.x[6:]):
                            polar_res = np.append(polar_res,i)
                        results_polar = scipy.optimize.minimize(func_dual, polar_res, args=(metric_function, dual_craters, craters_world, K, dual_method, W, continuous, Pm_c), method="SLSQP", jac=None, hess=None, hessp=None, bounds=bounds_x0, constraints=(), tol=None, callback=None, options={"disp":False, "ftol":1e-10})
                        print(results_polar.fun, fval)
                        if results_polar.fun < fval:
                            results = results_polar
                            inverse_true = True
                    
                        scales_est =results.x[6:]#results[0][6:]
                        weights_est = [get_weight(dual_method,s) for s in scales_est]

                        # results = scipy.optimize.minimize(func_dual_optimise_pose, x0[:6], args=(metric_function, dual_craters, craters_world, K, dual_method, W, continuous, Pm_c), method="SLSQP", jac=None, hess=None, hessp=None, bounds=bounds_x0, constraints=(), tol=None, callback=None, options={"disp":False, "ftol":1e-10})
                        
                        # scales_est = [-30, -30, -30, 30, 30, -30, -30, -30, -30, -30, 30, -30]
                        # x0 = [-180.00140261558957, -30.002087444593293, 179.99589612666364, 5775.367857955016, 3331.609339275741, 1833838.973300957]
                        # results = scipy.optimize.minimize(func_dual_optimise_pose, x0, args=(metric_function, dual_craters, scales_est, craters_world, K, dual_method, W, continuous, Pm_c), method="SLSQP", jac=None, hess=None, hessp=None, bounds=bounds_x0[:6], constraints=(), tol=None, callback=None, options={"disp":False, "ftol":1e-10})
                     
              
                        weights_est = [get_weight(dual_method,s) for s in scales_est]

                        break

                    # Adjust search bounds
                    # TODO: (Sofia) I don't know if this makes sense - fval should always be lower with higher scale (?).
                    current_min_scale = current_min_scale + current_min_scale/2
                    current_max_scale = current_max_scale + current_max_scale/2
        
                    fval_prev = fval
                    x0[:6] = results.x[:6]#results[0][:6]

                    x0[6:] = results.x[6:]#results[0][6:]
                    for i in range(len(craters_world)):
                        if round(x0[6+i]) < 0:
                            x0[6+i] = current_min_scale
                        else:
                            x0[6+i] = current_max_scale
                        # x0[6:] = [-current_max_scale, current_max_scale, -current_max_scale, current_max_scale]
                        bounds_x0[6+i] = (current_min_scale, current_max_scale) 
            else:
                results = scipy.optimize.minimize(func_dual, x0, args=(metric_function, dual_craters, craters_world, K, dual_method, W, continuous, Pm_c), method="SLSQP", jac=None, hess=None, hessp=None, bounds=bounds_x0, constraints=(), tol=None, callback=None, options={"disp":False, "ftol":1e-10})
            
            # errs.append(results[1])
            # res_eulers.append([results[0][0], results[0][1], results[0][2]])
            # rm_ests.append([results[0][3], results[0][4], results[0][5]])
            # opt_res.append(results) 
            error = results.fun
            errs.append(error)
            res_eulers.append(results.x[:3])
            rm_ests.append(results.x[3:6])
            opt_res.append(results.x) 

            pose_est = results.x[:6] #results[0][:6]
            dual_weights_est = np.around(weights_est)
            dual_weights.append(dual_weights_est)

        else:
            if m_estimator != None:
                err = func(x0, metric_function, image_conics, craters_world, K, W)
                prev_err = 0
                it = 0
                max_it = 25
                e = 0.01 # IRLS termination condition

                while (abs(prev_err - err) > e and it <= max_it):
                    prev_err = err
                    if euler_bound == 0:
                        results = scipy.optimize.fmin_l_bfgs_b(func_optimise_position, x0[3:], fprime=None, args=(metric_function, image_conics, craters_world, K, x0[:3], W, continuous, Pm_c), approx_grad=1, bounds=bounds_x0[3:], m=10, factr=factr_, pgtol=pgtol_, epsilon=epsilon_, iprint=-1, maxfun=maxfun_, maxiter=maxiter_, disp=False, callback=None, maxls=maxls_)
                        res_euler = [x0[:3]]
                        rm_est = [results[0][0], results[0][1], results[0][2]]
                    else:
                        results = scipy.optimize.fmin_l_bfgs_b(func, x0, fprime=None, args=(metric_function, image_conics, craters_world, K, W, continuous, Pm_c), approx_grad=1, bounds=bounds_x0, m=10, factr=factr_, pgtol=pgtol_, epsilon=epsilon_, iprint=-1, maxfun=maxfun_, maxiter=maxiter_, disp=False, callback=None, maxls=maxls_)
                        res_euler = [results[0][0], results[0][1], results[0][2]]
                        rm_est = [results[0][3], results[0][4], results[0][5]]
                    err = results[1]
                    opt_res.append(results)    
                    errs.append(err)
                    res_eulers.append(res_euler)
                    rm_ests.append(rm_est)
                    opt_res.append(results)

                    # results = scipy.optimize.fmin_l_bfgs_b(func, x0, fprime=None, args=(metric_function, image_conics, craters_world, K, W), approx_grad=1, bounds=bounds_x0, m=10, factr=factr_, pgtol=pgtol_, epsilon=epsilon_, iprint=-1, maxfun=maxfun_, maxiter=maxiter_, disp=False, callback=None, maxls=maxls_)

                    # errs.append(err)
                    # res_eulers.append(res_euler)
                    # rm_ests.append(rm_est)
                    # opt_res.append(results)
                    
                    Pm_c_estimated = projection_matrix_from_euler_and_position(K, res_euler, rm_est)
                    

                    # Update weighted matrix using m-estimators - method Tukey.
                    W = m_estimators_weighted_matrix(intensity_func, image_conics, craters_world, Pm_c_estimated, estimator = m_estimator, c = tuning_const)
                    it += 1

                # for i in correctly_matched_crater_indices:
                #     print(intensity_func(image_conics[i], craters_world[i],Pm_c),",",1)

                # for i in switched_crater_indices:
                #     print(intensity_func(image_conics[i], craters_world[i],Pm_c),",",0)

                for i in correctly_matched_crater_indices:
                    if np.diag(W)[i] == 0:
                        print("WARNING: correctly marked crater marked incorrect")
                        print("i:",i)
                        print("W[i]:",np.diag(W)[i])
                        print("error i:", intensity_func(image_conics[i], craters_world[i],Pm_c_estimated))
                        fn += 1
                for i in switched_crater_indices:
                    if np.diag(W)[i] != 0:
                        print("WARNING: incorrectly matched crater passed")
                        print("i:",i)
                        print("W[i]:",np.diag(W)[i])
                        print("error i:", intensity_func(image_conics[i], craters_world[i],Pm_c_estimated))
                        fp += 1

            else:
                # bounds_x0 = [(None,None),(None,None),(None,None),(None,None),(None,None),(None,None)]
                # TODO: same for m-est
                if euler_bound == 0:
                    results = scipy.optimize.fmin_l_bfgs_b(func_optimise_position, x0[3:], fprime=None, args=(metric_function, image_conics, craters_world, K, x0[:3], W, continuous, Pm_c), approx_grad=1, bounds=bounds_x0[3:], m=10, factr=factr_, pgtol=pgtol_, epsilon=epsilon_, iprint=-1, maxfun=maxfun_, maxiter=maxiter_, disp=False, callback=None, maxls=maxls_)
                    res_eulers.append([x0[:3]])
                    rm_ests.append([results[0][0], results[0][1], results[0][2]])
                else:
                    results = scipy.optimize.fmin_l_bfgs_b(func, x0, fprime=None, args=(metric_function, image_conics, craters_world, K, W, continuous, Pm_c), approx_grad=1, bounds=bounds_x0, m=10, factr=factr_, pgtol=pgtol_, epsilon=epsilon_, iprint=-1, maxfun=maxfun_, maxiter=maxiter_, disp=False, callback=None, maxls=maxls_)
                    res_eulers.append([results[0][0], results[0][1], results[0][2]])
                    rm_ests.append([results[0][3], results[0][4], results[0][5]])
                opt_res.append(results[0])    
                errs.append(results[1])
                    

                # results = scipy.optimize.minimize(func, x0, args=(metric_function, image_conics, craters_world, K, W), method="Powell", jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
                # # print(results)
                # errs.append(results.fun)
                # res_eulers.append([results.x[0], results.x[1], results.x[2]])
                # rm_ests.append([results.x[3], results.x[4], results.x[5]])
                # # dual_weights_est = results[0][6:]
                # opt_res.append(results)    

        min_error_index = np.argmin(errs)
        err = errs[min_error_index]
        res_euler = res_eulers[min_error_index]
        rm_est = rm_ests[min_error_index]
        min_res = opt_res[min_error_index]
        pose_est = min_res[:6]
        
        if pose_method == "dual_PnP_linear" or pose_method == "dual_PnP_sigmoid" or pose_method == "dual_PnP_sigmoid_permutations":
            min_dual_weights = dual_weights[min_error_index]

        Pm_c_estimated = projection_matrix_from_euler_and_position(K, res_euler, rm_est)

        
        if show_all_combinations and (pose_method == "dual_PnP_linear" or pose_method == "dual_PnP_sigmoid" or pose_method == "dual_PnP_sigmoid_permutations" or pose_method == "maass_cubic_pnp" or pose_method == "maass_cubic_pnp_optimised"):
            current_max_scale = 30
            current_min_scale = -current_max_scale
            ground_truth_dual_craters = dual_crater_from_imaged_ellipse(image_conics, craters_world, K, Pm_c, continuous = True)
            ## Find the weights of the dual circles that best match (based on angular and euclidean distance) to the world craters.
            closest_dual_weights, combinations = minimize_3D_crater_distance(dual_craters, craters_world)
            singular_closest_dual_scales = []
            for w in closest_dual_weights:
                if w[0]:
                    singular_closest_dual_scales.append(current_max_scale)
                else:
                    singular_closest_dual_scales.append(current_min_scale)
            noise_weights_est = np.around([get_weight("sigmoid",s) for s in singular_closest_dual_scales])

            scaled_combinations = []
            for i, combination in enumerate(combinations):
                scales = []
                for j in range(len(combination)):
                    if combination[j] == 1:
                        scales.append(current_max_scale)
                    else:
                        scales.append(current_min_scale)
                scaled_combinations.append(scales)

            ## The following code chooses the combination that produces the best 
            fvals_e = []
            surf_errs = []
            pos_errs = []
            for combination in scaled_combinations:
                combination_results = scipy.optimize.minimize(func_dual_optimise_pose, pose_est, args=(dual_pnp_reprojection_error, dual_craters, combination, craters_world, K, "sigmoid", W, continuous, Pm_c), method="SLSQP", jac=None, hess=None, hessp=None, bounds=bounds_x0[:6], constraints=(), tol=None, callback=None, options={"disp":False, "ftol":1e-10})
                fvals_e.append(combination_results.fun)
                Pm_c_estimated_test = projection_matrix_from_euler_and_position(K, combination_results.x[:3], combination_results.x[3:6])
                surface_observation_error = get_surface_observation_error(K, Pm_c, Pm_c_estimated_test, scale, min_offset)
                surf_errs.append(surface_observation_error)
                abs_diff_m, _, _, _ = get_position_offset(K, Pm_c_estimated_test, Pm_c, min_offset, scale)
                pos_errs.append(abs_diff_m)
            print("Img:",str(file_name))
            print()
            print("Best combinations:")
            print("best pose with comb and fval:",min(pos_errs), combinations[np.argmin(pos_errs)],fvals_e[np.argmin(pos_errs)])
            print("best surf er with comb and fval:",min(surf_errs), combinations[np.argmin(surf_errs)],fvals_e[np.argmin(surf_errs)])

            combination_results = scipy.optimize.minimize(func_dual_optimise_pose, pose_est, args=(dual_pnp_reprojection_error, dual_craters, singular_closest_dual_scales, craters_world, K, "sigmoid", W, continuous, Pm_c), method="SLSQP", jac=None, hess=None, hessp=None, bounds=bounds_x0[:6], constraints=(), tol=None, callback=None, options={"disp":False, "ftol":1e-10})
            Pm_c_estimated_test = projection_matrix_from_euler_and_position(K, combination_results.x[:3], combination_results.x[3:6])
            surface_observation_error = get_surface_observation_error(K, Pm_c, Pm_c_estimated_test, scale, min_offset)
            abs_diff_m, _, _, _ = get_position_offset(K, Pm_c_estimated_test, Pm_c, min_offset, scale)
            print()
            print("Minimised combination:")
            print("minimised combination",noise_weights_est)
            print("pose error under minimised distances and fval:",abs_diff_m,combination_results.fun)
            print("surf error under minimised distances and fval:",surface_observation_error,combination_results.fun)

            # print()
            best_comb = scaled_combinations[np.argmin(fvals_e)]
            best_comb = [get_weight(dual_method,s) for s in best_comb]
            # print("Pose est with different combinations:\nOptimised fval vs min fval vs diff:",results.fun, min(fvals_e),min(fvals_e)-results.fun,np.around(best_comb),"\n")

            
            fvals = []
            
            for combination in scaled_combinations:
                fval_true_pose = func_dual_optimise_pose(np.hstack((Tm_c_euler,rm)),metric_function, dual_craters, combination, craters_world, K, "sigmoid", W, continuous, Pm_c)
                fvals.append(fval_true_pose)
            best_comb = scaled_combinations[np.argmin(fvals)]
            best_comb = [get_weight(dual_method,s) for s in best_comb]


        

    finish_time = time.perf_counter()
    runtime = finish_time-start_time

    # Compute the position offset.
    abs_diff_m, X_diff_m, Y_diff_m, Z_diff_m = get_position_offset(K, Pm_c_estimated, Pm_c, min_offset, scale)
    position_offsets_m = [abs_diff_m, X_diff_m, Y_diff_m, Z_diff_m]

    # Compute the orientation eror.
    rotation_angle_difference_error = angle_of_difference_rotation(K, Pm_c_estimated, Pm_c)*180/math.pi

    # Plot the projected craters on the image.
    metric_dir = metric+"_metric/"
    if not os.path.isdir(write_dir + noise_directory + metric_dir):
        os.makedirs(write_dir + noise_directory + metric_dir)
    # Noisy craters.
    
    # Project noisy crater matches.
    incorrectly_matched_craters = []
    incorrectly_matched_crater_indices = []
    correctly_matched_craters = []
    number_of_discarded_craters = 0
    for c in range(len(craters_world)):
        if np.diag(W)[c] == 0:
            number_of_discarded_craters += 1
        if c in switched_crater_indices:
            incorrectly_matched_craters.append(craters_world[c])
            incorrectly_matched_crater_indices.append(c)
        else:
            correctly_matched_craters.append(craters_world[c])

    # Store projected ellipses in a csv.
    proj_ellipse_csv_dir = write_dir+noise_directory+metric_dir+'projected_ellipse_csvs/'
    if not os.path.isdir(proj_ellipse_csv_dir):
        os.makedirs(proj_ellipse_csv_dir)
    csv_file = proj_ellipse_csv_dir+str(file_name)+'.csv'
    csv_f = open(csv_file, "w")
    csv_f.write("(x_c, y_c, a, b, phi) ground truth projected crater location, noisy crater projected location, estimated crater projected location, incorrectly matched crater, t/f\n")
    for i in range(len(craters_world)):
        write_conic_to_projected_ellipse_csv_no_return(conic_from_crater(craters_world[i], Pm_c), csv_f) # ground truth ellipse
        csv_f.write(", ")
        write_conic_to_projected_ellipse_csv_no_return(image_conics[i], csv_f) # noisy ellipse
        csv_f.write(", ")
        write_conic_to_projected_ellipse_csv_no_return(conic_from_crater(craters_world[i], Pm_c_estimated), csv_f) # estimated ellipse
        csv_f.write(", ")
        if i in switched_crater_indices:
            csv_f.write("1\n")
        else:
            csv_f.write("0\n")

    switched_crater_indices.sort()
    incorrectly_matched_crater_indices.sort()

    position_offset_error = observed_surface_width_metric(position_offsets_m[0], K, Pm_c, image)

    world_camera_surface_intersection_points = position_offset_error[3]
    position_offset_error = get_position_offset_error(rm, position_offsets_m[0])

       
    surface_observation_error_pixel_resolution = get_surface_observation_error_as_function_of_pixel_resolution(craters_world, Pm_c, Pm_c_estimated, image_to_world_resolution = 100)
    surface_observation_error = get_surface_observation_error(K, Pm_c, Pm_c_estimated, scale, min_offset)


    # Project craters.
    img_dir = noise_directory+metric_dir+'projection_img_'+str(file_name)+'.png'

    # Project the world crater rim in green.
    # project_craters(correctly_matched_craters, Pm_c, image, write_dir, img_dir, (22, 166, 15)) #green
    # #Project the image conic in orange.
    # project_conics(image_conics, image, write_dir, img_dir, (5, 153,245),False, size=4) #orange
    image = show_crater_centres(craters_world, Pm_c, image, (22, 255, 15),size=4) # Project true crater centre.
    # cv2.imwrite(write_dir+img_dir, image)

    # Project dual crater centres.
    if show_all_combinations:
        if pose_method == "dual_PnP_linear" or pose_method == "dual_PnP_sigmoid" or pose_method == "dual_PnP_sigmoid_permutations":
            for i, dual_crater in enumerate(dual_craters):
                dual_Pm_c = np.hstack((K,np.array([0,0,0]).reshape((3,1))))
                if round(weights_est[i]) != round(noise_weights_est[i]):
                    x_c, y_c, _, _, _ = conic_matrix_to_ellipse(image_conics[i])
                    image = cv2.circle(image, (int(x_c), int(y_c)), 10, (0,0,255), 0)
                    # print("wrong duals for image:",img_dir,
                    #       "\n projected dual centres:",proj_centre_loc_dual(dual_crater[0],dual_Pm_c),proj_centre_loc_dual(dual_crater[1],dual_Pm_c),
                    #       "\ndiff:",np.linalg.norm(proj_centre_loc_dual(dual_crater[0],dual_Pm_c)-proj_centre_loc_dual(dual_crater[1],dual_Pm_c)),
                    #       "\nprojected crater center (true pose vs estimated pose):",proj_centre_loc_dual(craters_world[i],Pm_c),proj_centre_loc_dual(craters_world[i],Pm_c_estimated),"\n")
                # Project the ground truth dual crater locations.
                if round(noise_weights_est[i]) == 1:
                    project_dual_crater_centres([ground_truth_dual_craters[i][1]], dual_Pm_c, image, write_dir, img_dir, (235, 161, 52), size =2) # cyan dual crater not chosen
                    project_dual_crater_centres([ground_truth_dual_craters[i][0]], dual_Pm_c, image, write_dir, img_dir, (153, 52, 235), size =2) # pink dual crater is chosen
                else:
                    project_dual_crater_centres([ground_truth_dual_craters[i][0]], dual_Pm_c, image, write_dir, img_dir, (235, 161, 52), size =2) # cyan dual crater not chosen
                    project_dual_crater_centres([ground_truth_dual_craters[i][1]], dual_Pm_c, image, write_dir, img_dir, (153, 52, 235), size =2) # pink dual crater is chosen
                # Project selected dual crater 
                if round(weights_est[i]) == 1:
                    project_dual_crater_centres([dual_crater[1]], dual_Pm_c, image, write_dir, img_dir, (255, 255, 0), size =2) # cyan dual crater not chosen
                    project_dual_crater_centres([dual_crater[0]], dual_Pm_c, image, write_dir, img_dir, (255, 0, 255), size =2) # pink dual crater is chosen
                else:
                    project_dual_crater_centres([dual_crater[0]], dual_Pm_c, image, write_dir, img_dir, (255, 255, 0), size =2)
                    project_dual_crater_centres([dual_crater[1]], dual_Pm_c, image, write_dir, img_dir, (255, 0, 255), size =2)
                project_dual_crater_centres([craters_world[i]], Pm_c_estimated, image, write_dir, img_dir, (0, 255, 255), size =2)
            project_craters(craters_world, Pm_c_estimated, image, write_dir, img_dir, (0, 255, 255))
            
    else:    
        project_conics(image_conics, image, write_dir, img_dir, (0, 0, 255),True) #red
        project_craters(correctly_matched_craters, Pm_c, image, write_dir, img_dir, (20, 255, 20)) #green 
        project_craters(craters_world, Pm_c_estimated, image, write_dir, img_dir, (0, 255, 255))#, fill=m_estimator, intensities=intensities) #yellow
    # # project_craters(incorrectly_matched_craters, Pm_c, image, write_dir, noise_directory+metric_dir+'projection_img_'+str(file_name)+'.png', (255, 0, 0)) #magenta 

    # Uncomment if you want the line of best fit.
    # cv2.line(image, (p1[0],p1[1]), (p2[0],p2[1]), (255,255,255), 7) 

    # Write error results on image.
    image = cv2.putText(image, "observed surface error (m): "+str(round(surface_observation_error,2)), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    image = cv2.putText(image, "position error (m): "+str(round(position_offsets_m[0],2))+"m", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    image = cv2.putText(image, "angular error (deg): "+str(round(rotation_angle_difference_error,2))+"deg", (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    # image = cv2.putText(image, "favl : "+str(err), (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imwrite(write_dir+img_dir, image)

    # NOTE: ground truth error is subjective to which of the two dual circles you weight (if there is noise, neither of the circles are "correct")
    W = np.identity(len(craters_world)) # weighted matrix.
    if pose_method == "dual_PnP_linear" or pose_method == "dual_PnP_sigmoid" or pose_method == "dual_PnP_sigmoid_permutations" or pose_method == "maass_cubic_pnp" or pose_method == "maass_cubic_pnp_optimised":
        # print("w true:", singular_closest_dual_scales)
        # print("w est:", min_dual_weights )
        if show_all_combinations:
            gnd_truth_error = 1 #metric_function(dual_craters, singular_closest_dual_scales, craters_world, Pm_c, K, dual_method, W)
        else:
            gnd_truth_error = 1
        # TODO: (Sofia) change
        # gnd_truth_error = dual_pnp_init_metric(image_conics, craters_world, Pm_c, W)

        # Get the returned surface energy from the selected duals.
        true_duals = []
        dual_projection_matrix = np.hstack((K,np.array([0,0,0]).reshape((3,1))))
        for i,w in enumerate(np.around(weights_est)):
            if w == 1:
                true_duals.append(dual_craters[i][0])
            else:
                true_duals.append(dual_craters[i][1])
        if scale != 1:
            selected_energy = surface_energy_from_conics(true_duals, image_conics)
        else:
            crater_distances = []
            for crater in craters_world:
                crater_distances.append(np.linalg.norm([crater.X, crater.Y, crater.Z]))
            scale_test = max(crater_distances)
            
            offset = 0
            scaled_dual_craters_world = []
            for i, crater in enumerate(true_duals):
                if not is_pangu:
                    scaled_dual_craters_world.append(Crater_w_scaled((crater.X - offset)/scale_test, (crater.Y - offset)/scale_test, (crater.Z - offset)/scale_test, (crater.r)/scale_test, (crater.r)/scale_test, crater.phi,"",is_pangu, crater.norm))
                else:
                    scaled_dual_craters_world.append(Crater_w_scaled((crater.X - offset)/scale_test, (crater.Y - offset)/scale_test, (crater.Z - offset)/scale_test, (crater.r)/scale_test, (crater.r)/scale_test, crater.phi,"crater.id",is_pangu=is_pangu))
            
            k_extrinsic_matrix_ground_truth = np.dot(np.linalg.inv(K),dual_projection_matrix)
            R_gt = k_extrinsic_matrix_ground_truth[0:3,0:3]
            ground_truth_world_position = -1*np.dot(np.linalg.inv(R_gt), k_extrinsic_matrix_ground_truth[:,3])
            ground_truth_world_position = np.array([v/scale_test + min_offset for v in ground_truth_world_position])

            ground_truth_world_position = np.dot((R_gt), -1*ground_truth_world_position).reshape((3,1))
            k_extrinsic_matrix_ground_truth = np.dot((K),np.hstack((R_gt,ground_truth_world_position)))

            selected_energy = surface_energy_from_conics(scaled_dual_craters_world, image_conics)
    else:
        gnd_truth_error = metric_function(image_conics, craters_world, Pm_c, W)
        selected_energy = true_energy

    print()
    print("...", file_name)
    print("      metric:",metric)
    print("      surface observation error (m): ",surface_observation_error)
    print("      surface observation error using pix. res. (m): ",surface_observation_error_pixel_resolution)
    print("      position error (abs, x, y, z) (m): ", position_offsets_m)
    print("      angular error (deg): ", rotation_angle_difference_error)
    print("      objective value (optimised pose): ", err)
    print("      objective value (ground truth pose): ", gnd_truth_error)
    print("      objective value (optimised pose) > objective value (ground truth pose): ", err > gnd_truth_error)
    print("      number of craters correctly matched: ",len(craters_world)-len(switched_crater_indices))
    print("      number of craters used by m-estimators: ",str(len(craters_world)-number_of_discarded_craters)+"/"+str(len(craters_world)))
    print("      crater distribution std: ",crater_appearence_std)
    print("      average crater distances from line of best fit: ",np.average(distances_from_line_of_best_fit))
    print("      std crater distances from line of best fit: ",np.std(distances_from_line_of_best_fit))
    print("      number of false positives: ",fp)
    print("      number of false negatives: ",fn)
    print("      average crater size pix^2",np.average(np.array(crater_sizes)))
    print("      median crater size pix^2",np.median(np.array(crater_sizes)))
    print("      true curvature energy:  ",true_energy)
    # print("      optimiser results:",min_res)
    if pose_method == "dual_PnP_linear" or pose_method == "dual_PnP_sigmoid" or pose_method == "dual_PnP_sigmoid_permutations" or pose_method == "maass_cubic_pnp" or pose_method == "maass_cubic_pnp_optimised":
        print("      estimated curvature energy:  ",selected_energy)
        print("      inversed: ",inverse_true)
        # if show_all_combinations:
        #     # print("      Optimised fval vs min fval vs diff:",results.fun, min(fvals),min(fvals)-results.fun)
        #     print("   Estimated scales:         ",(results.x[6:]), 
        #         "\n   Best noise scales:              ",singular_closest_dual_scales,
        #         "\n\n   Estimated rounded weights:",np.around(weights_est),
        #         "\n   Best noise rounded weights:     ",np.around(noise_weights_est),
        #         "\n   fval, true fval, true lin fval",results.fun,func_dual_optimise_pose(np.hstack((Tm_c_euler,rm)),metric_function, dual_craters, singular_closest_dual_scales, craters_world, K, "sigmoid", W, continuous, Pm_c),func_dual_optimise_pose(np.hstack((Tm_c_euler,rm)),metric_function, dual_craters, np.around(noise_weights_est), craters_world, K, "linear", W, continuous, Pm_c),"\n")
        #     print("    TERMINATION:\nScale bounds:",[current_min_scale, current_max_scale],"\nFvalPrev - Fval:", fval_prev,"-",fval,"=",fval_prev-fval,"\nIterations:",it,"\nResults:",results,"\n")
        # else:
        print("weights:",np.around(weights_est))

    print("Program finished in {} seconds".format(runtime))
    print("---")

    # Save plotted camera and image positions from estimated and ground truth calibration data.
    # all_craters_world = get_craters_world(os.path.abspath(dir+craters_world_dir+crater_world_filenames[0]))
    # plot_camera_pose(K, Pm_c_estimated, Pm_c, all_craters_world, craters_world, dir, write_dir, noise_pixel_offset, euler_bound, position_bound, file_name, metric, world_camera_surface_intersection_points, extrinsics_visualisation)
    funcalls = 0
    nit = 0
    # if min_res is not None:
    #     funcalls = min_res[2]['funcalls']
    #     nit = min_res[2]['nit']
    m_est_iterations = 0
    if m_estimator != None:
        try:
            m_est_iterations = it
        except:
            m_est_iterations = 0

    pose_dict = {
        "surface_observation_error_m" : [surface_observation_error],
        "surface_observation_error_pixel_resolution_m" : [surface_observation_error_pixel_resolution],
        "position_offset_m_abs" : [position_offsets_m[0]],
        "position_offset_m_x" : [position_offsets_m[1]],
        "position_offset_m_y" : [position_offsets_m[2]],
        "position_offset_m_z" : [position_offsets_m[3]],
        "angular_error" : [rotation_angle_difference_error], 
        "objective_function_error" : [err],
        "gnd_truth_objective_function_error" : [gnd_truth_error],
        "total_number_of_craters" : [len(craters_world)],
        "number_of_correct_craters_avaliable" : [len(craters_world)-len(switched_crater_indices)],
        "average_crater_diameter_m" : [np.average(np.array(crater_diameters_m))],
        "median_crater_diameter_m" : [np.median(np.array(crater_diameters_m))],
        "std_crater_diameter_m" : [np.std(np.array(crater_diameters_m))],
        "projected_average_crater_size_pix^2" : [np.average(np.array(crater_sizes))],
        "projected_median_crater_size_pix^2" : [np.median(np.array(crater_sizes))],
        "projected_std_crater_size_pix^2" : [np.std(np.array(crater_sizes))],
        "crater_appearence_std" : [crater_appearence_std],
        "average_distances_from_line_of_best_fit" : [np.average(distances_from_line_of_best_fit)],
        "sum_distances_from_line_of_best_fit" : [np.sum(distances_from_line_of_best_fit)],
        "fp" : [fp],
        "fn" : [fn],
        "file_name" : [file_name],
        "runtime" : [runtime],
        "funcalls" : [funcalls],
        "nit" : [nit],
        "m_est_iterations": [m_est_iterations],
        "inversed" : [int(inverse_true)],
        "true_energy" : [true_energy],
        "dual_selection_energy" : [selected_energy],
    }

    pose_result = pose_dict

    return pose_result

    
    
    
    
    
    
    
    