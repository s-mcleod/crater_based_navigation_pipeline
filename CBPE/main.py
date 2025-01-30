import argparse
import cv2
import math
from mpmath import mp
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Circle, PathPatch
from numpy import linspace
import mpl_toolkits.mplot3d.art3d as art3d
import os
import random

from multiprocessing import Pool
import math

from src.camera_pose_visualisation import *
from src.get_data import *
from src.metrics import *
from src.extrinsics import *
from src.image_display import *
from src.pose_estimation import *
from src.set_data import *

def pose_estimation_pool_process(
        i, \
        image_files, \
        ground_truth_poses, \
        dir, \
        img_folder, \
        craters_world_dir, \
        crater_world_filenames, \
        craters_cam_dir, \
        crater_cam_filenames, \
        add_noise, \
        sig, \
        position_bound, \
        pose_method, \
        num_craters, \
        min_craters_available, \
        K, \
        distCoeffs, \
        use_scale, \
        max_noise_sigma_pix, \
        write_dir, \
        gauss_pixel_noise, \
        extrinsics_visualisation, \
        metric, \
        euler_bound, \
        proportion_incorrect_crater_matches, \
        m_estimator, \
        tuning_const, \
        factr, \
        epsilon, \
        pgtol, \
        maxfun, \
        maxiter, \
        maxls, \
        # degrees_off_nadir_dict, \
        continuous, \
        is_pangu, \
        propagated_position = None, \
        ):

    # time.sleep(random.random())
    # for i in range(loop_range[0], loop_range[1], step):
    file_name = image_files[i][:-4]

    # Get ground truth pose.
    ground_truth_pose = ground_truth_poses[i]

    # Get the corresponding image.
    image = get_image(os.path.abspath(dir+img_folder+image_files[i]))

    # Get the sets of 3D and 2D crater points
    craters_world = get_craters_world(os.path.abspath(dir+craters_world_dir+crater_world_filenames[i]), is_pangu)
    craters_cam = get_craters_cam(os.path.abspath(dir+craters_cam_dir+crater_cam_filenames[i]), add_noise, 0, sig)

    if (crater_world_filenames[i] != crater_cam_filenames[i]):
        print("Imaged crater and matched crater files don't align:\n",os.path.abspath(dir+craters_world_dir+crater_world_filenames[i]),"\n",os.path.abspath(dir+craters_cam_dir+crater_cam_filenames[i]))

    for c in range(len(craters_world)):
        if craters_world[c].id != craters_cam[c].id:
            print(os.path.abspath(dir+craters_world_dir+crater_world_filenames[i]),"\n",os.path.abspath(dir+craters_cam_dir+crater_cam_filenames[i]))
            print("ID's ARE NOT THE SAME")


    # Scale data by scaling ground truth position, position bounds, and 3D craters.
    min_offset = 0
    scale = 1
    scaled_ground_truth_pose = ground_truth_pose
    scaled_craters_world = craters_world
    scaled_position_bound = position_bound
    if (use_scale):
        scaled_ground_truth_pose, scaled_craters_world, scaled_position_bound, scaled_propagated_position, min_offset, scale = set_scaled_selenographic_data(ground_truth_pose, craters_world, position_bound, propagated_position, is_pangu=is_pangu)

    # Get ground truth position.
    k_extrinsic_matrix_ground_truth, angle_off_nadir_deg = init_projection_matrix(ground_truth_pose, K, PANGU=is_pangu)
    scaled_k_extrinsic_matrix_ground_truth, angle_off_nadir_deg = init_projection_matrix(scaled_ground_truth_pose, K, PANGU=is_pangu)


    # Estimate the position given the ground truth attitude.
    pose_results_dict = pose_estimation(\
            pose_method,\
            scaled_craters_world, \
            craters_world, \
            craters_cam, \
            num_craters, \
            min_craters_available, \
            scaled_k_extrinsic_matrix_ground_truth, \
            k_extrinsic_matrix_ground_truth, \
            K, \
            distCoeffs, \
            image, \
            sig, \
            max_noise_sigma_pix, \
            dir, \
            write_dir, \
            gauss_pixel_noise, \
            file_name, \
            add_noise, \
            extrinsics_visualisation, \
            craters_world_dir, \
            crater_world_filenames, \
            metric, \
            euler_bound, \
            scaled_position_bound, \
            position_bound, \
            proportion_incorrect_crater_matches, \
            min_offset, \
            scale, \
            is_pangu, \
            m_estimator, \
            tuning_const, \
            factr_=factr, \
            epsilon_=epsilon, \
            pgtol_=pgtol, \
            maxfun_=maxfun, \
            maxiter_=maxiter, \
            maxls_=maxls,\
            seed=i,\
            continuous=continuous,
            propagated_position=scaled_propagated_position)
    # pose_results_dict["degree_off_nadir"] = [degrees_off_nadir_dict[file_name]]
    return (pose_results_dict)

def main():

    parser = argparse.ArgumentParser(description='Process files for crater projection.')
    parser.add_argument('dir') 
    parser.add_argument('craters_world_dir')
    parser.add_argument('craters_cam_dir')
    parser.add_argument('pangu_flight_file')
    parser.add_argument('img_folder')
    parser.add_argument('calibration_file')
    # parser.add_argument('degrees_off_nadir_file')
    parser.add_argument('--pose_method', type=str, help="options(1): [non_coplanar_conics, 6DoF_pnp_unbounded, 6DoF_pnp_bounded, 6DoF_gaussian_angle_unbounded, 6DoF_gaussian_angle_bounded, 6DoF_euclidean_distance_bounded, 6DoF_wasserstein_distance_unbounded, 6DoF_wasserstein_distance_bounded, 6DoF_level_set_based_unbounded, 6DoF_level_set_based_bounded, 6DoF_level_set_based_global, 6DoF_gaussian_angle_global, 6DoF_ellipse_distance_init_gaussian_angle] default: [pnp]")  # on/off flag 
    parser.add_argument('--extrinsics_visualisation',action='store_true')  # on/off flag 
    parser.add_argument('--add_noise',action='store_true')  # on/off flag 
    parser.add_argument('--m_estimator',action='store_true')  # on/off flag 
    parser.add_argument('--scale',action='store_true')  # on/off flag 
    parser.add_argument("--continuous",action='store_true') # on only if you want to project and add noise to the ground truth craters NOT the detected craters
    parser.add_argument("--not_pangu",action='store_true') # on only if you want to project and add noise to the ground truth craters NOT the detected craters
    parser.add_argument("--propagated_position", type=str)
    args = parser.parse_args()

    # Input.
    dir = args.dir
    craters_world_dir = args.craters_world_dir
    craters_cam_dir = args.craters_cam_dir
    pangu_flight_file = args.pangu_flight_file
    img_folder = args.img_folder
    calibration_file = args.calibration_file
    # degrees_off_nadir = args.degrees_off_nadir_file
    pose_method = args.pose_method
    extrinsics_visualisation = args.extrinsics_visualisation
    add_noise = args.add_noise
    use_m_estimators = args.m_estimator
    use_scale = args.scale
    continuous = args.continuous
    is_pangu = not args.not_pangu
    propagated_position_dir = args.propagated_position

    # Create a folder for the pose method.
    pose_method_dir = pose_method+"/"
    if use_m_estimators:
        pose_method_dir = pose_method+"_with_incorrect_matches/"
    if not os.path.isdir(dir+pose_method_dir):
        os.makedirs(dir+pose_method_dir)
    write_dir = dir+pose_method_dir

    # Metric to determine position error.
    metric = "height_to_surface"

    projection_dir = "projection_images/"
    if not os.path.isdir(write_dir+projection_dir):
        os.makedirs(write_dir+projection_dir)
    
    # Get input.
    ground_truth_poses = get_camera_poses(os.path.abspath(dir+pangu_flight_file))
    crater_world_filenames = get_files_in_dir(os.path.abspath(dir+craters_world_dir),"txt")
    crater_world_filenames.sort(key=lambda x: int(x[:-4]))
    crater_cam_filenames = get_files_in_dir(os.path.abspath(dir+craters_cam_dir),"txt")
    crater_cam_filenames.sort(key=lambda x: int(x[:-4]))
    image_files = get_files_in_dir(os.path.abspath(dir+img_folder), "png")
    image_files.sort(key=lambda x: int(x[:-4]))
    if propagated_position_dir:
        propagated_position_filenames = get_files_in_dir(os.path.abspath(dir+propagated_position_dir),"txt")
        propagated_position_filenames.sort(key=lambda x: int(x[:-4]))
        propagated_positions = []
        for file in propagated_position_filenames:
            f = open(os.path.abspath(dir+propagated_position_dir+file),"r")
            lines = f.readlines()
            lines = ([(re.split(r',\s*', line)) for line in lines])
            for line in lines:
                propagated_positions.append([float(i) for i in line])

 
    # Get the camera intrinsic matrix and distortion coeffients.
    K = get_intrinsic(os.path.abspath(dir+calibration_file))
    distCoeffs = np.array([])

    # Get the degrees off nadir for each associate file name.
    # deg_off_nadir_dict = file_name_deg_off_nadir_dict(dir+degrees_off_nadir)

    # Tightness of euler bounds 
    euler_bounds = [0] #0.01
    # Tightness of position bounds in metres. 
    position_bounds =  [6700] #6700
    
    # Parameter turning.
    pgtols=[1e-05] # best 1e-05
    factrs = [10] # best 10
    epsilons=[1e-08]  # best 1e-08
    maxfun=15000
    maxiter=15000
    maxls=20

    # The number of craters the optimiser will use.
    num_craters = 1000000
    # Selecting only images that have a minimum number of craters available - set to 0 to use the number of craters specified in num_craters_lists.
    min_craters_available = 3

    # Noisy crater matches.
    proportions_of_incorrect_crater_matches = [0]
    m_estimator = None
    if (use_m_estimators):
        proportions_of_incorrect_crater_matches = [0] #, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        m_estimator = "Tukey"
    tuning_const = 100 #[0.25] for simulated data # 200 for euler 5deg pos 10km no noise
    # print("m-estimator:",m_estimator)

    # Noisy crater detection.
    # We either set the random noise sampling sigma to be an ellipse's semi minor axis to be the max of noise*semiminor axis OR some maximum pixel offset
    noise = [0]
    max_noise_sigma_pix = 0
    if add_noise:
        # noise = [0.5]
        noise = [0.2]
        max_noise_sigma_pix = 1

    # Recording the average position offsets of the estimated and ground truth position vectors (add noise only).
    for proportion_incorrect_crater_matches in proportions_of_incorrect_crater_matches:
        # print("Proportion of incorrect matches:", proportion_incorrect_crater_matches)
        # print("Tuning constant:", tuning_const)          
        for sig in noise:
            # print("Perturbation:",sig)
            for euler_bound in euler_bounds:
                for position_bound in position_bounds:
                    # Output file.
                    all_stats_file = open(write_dir+"all_stats_"+str(sig)+"_"+str(proportion_incorrect_crater_matches)+"_"+str(m_estimator)+"_"+str(euler_bound)+"_"+str(position_bound)+".csv", "w")

                    # Output file.
                    f_performance_table = open(write_dir+"stats_"+str(proportion_incorrect_crater_matches)+"_"+str(sig)+"_"+str(num_craters)+"_"+str(m_estimator)+"_"+str(euler_bound)+"_"+str(position_bound)+".csv", "w")
                    f_performance_table.write("Euler_bounds,")
                    for euler_bound in euler_bounds:
                        f_performance_table.write(str(euler_bound)+",")
                    f_performance_table.write("\nPosition_bounds,")
                    for position_bound in position_bounds:
                        f_performance_table.write(str(position_bound)+",")
                    f_performance_table.write("\neuler bound (deg),position bound (m),avg surface observation error (m),avg pos error (m),avg angular error (deg),std surface observation error (m),std pos error (m),std angular error (deg),med surface observation error (m),med pos error (m),med angular error (deg),avg obj value,med obj value,avg obj value (gnd truth), average number of craters used, avg num false positives, avg num false negatives\n")

                    gauss_pixel_noise = projection_dir+str(proportion_incorrect_crater_matches)+"_"+str(sig)+"_"+str(num_craters)+"_"+str(euler_bound)+"_"+str(position_bound)+"_"+str(m_estimator)+"/"
                    if not os.path.isdir(write_dir+gauss_pixel_noise):
                        os.makedirs(write_dir+gauss_pixel_noise)

                    ### Image data restrictions ###
                    # Looping through all images.
                    loop_range = [0, len(ground_truth_poses)]
                    loop_range = [0,1500]
                    # loop_range = [0, 10]#len(ground_truth_poses)]
                    step = 1
                    # Limit dataset by number of images per deg. off nadir.
                    # TODO:(Sofia) Remove this constraint.
                    # limit_data = False # Set to false if you don't want to do this.
                    # file_numbers = [0, 0, 0, 0, 0, 0, 0]
                    # max_num_problem_instances_per_degree_off_nadir = 50
                    # deg_nad_inc = 10
                    # min_deg_off_nad = min(deg_off_nadir_dict.values())
                    # deg_nad_inc = 5
                    # file_numbers = [0]*14
                    
                    ###############################
                    
                    pose_results_dict = {}
                    # print("Crater noise proportion:", sig)
                    # print("Euler bound (deg):", euler_bound)
                    # print("Position bound (m):", position_bound)
                    for pgtol in pgtols:
                        # print("Pgtol:", pgtol)
                        for factr in factrs:
                            # print("Factr:", factr)
                            for epsilon in epsilons:
                                # print("Epsilon:", epsilon)
                                number_of_processed_images = 0
                                with Pool() as pool:
                                    args = []
                                    for i in range(loop_range[0], loop_range[1], step):
                                        # Make sure that the data we are processing has at least 3 craters.
                                        f = open(os.path.abspath(dir+craters_cam_dir+crater_cam_filenames[i]),"r")
                                        lines = f.readlines()[1:] #ignore the first line
                                        if len(lines) < 3:
                                            continue

                                        if propagated_position_dir:
                                            propagated_position = propagated_positions[i]
                                        else:
                                            propagated_position = None
                                        # if limit_data:
                                            
                                        #     ind = int((int(deg_off_nadir_dict[str(i)]) - min_deg_off_nad)/deg_nad_inc)
                                        #     if file_numbers[ind] >= max_num_problem_instances_per_degree_off_nadir:
                                        #         continue
                                        #     else:
                                        #         file_numbers[ind] += 1
                                        # if image_files[i] != "0.png":
                                        #     continue
                                        args.append((i, \
                                            image_files, \
                                            ground_truth_poses, \
                                            dir, \
                                            img_folder, \
                                            craters_world_dir, \
                                            crater_world_filenames, \
                                            craters_cam_dir, \
                                            crater_cam_filenames, \
                                            add_noise, \
                                            sig, \
                                            position_bound, \
                                            pose_method, \
                                            num_craters, \
                                            min_craters_available, \
                                            K, \
                                            distCoeffs, \
                                            use_scale, \
                                            max_noise_sigma_pix, \
                                            write_dir, \
                                            gauss_pixel_noise, \
                                            extrinsics_visualisation, \
                                            metric, \
                                            euler_bound, \
                                            proportion_incorrect_crater_matches, \
                                            m_estimator, \
                                            tuning_const, \
                                            factr, \
                                            epsilon, \
                                            pgtol, \
                                            maxfun, \
                                            maxiter, \
                                            maxls, \
                                            # deg_off_nadir_dict, \
                                            continuous, \
                                            is_pangu,
                                            propagated_position,\
                                                ))
                                    async_results = pool.starmap(pose_estimation_pool_process, args)

                                    for pose_result in async_results:
                                        if (len(pose_results_dict) == 0):
                                            pose_results_dict = pose_result
                                        else:
                                            for key in pose_results_dict:
                                                pose_results_dict[key].append(pose_result[key][0])
                                        number_of_processed_images += 1

                                for key in pose_results_dict:
                                    all_stats_file.write(key+",")
                                    for value in pose_results_dict[key]:
                                        all_stats_file.write(str(value)+",")
                                    all_stats_file.write("\n")

                                print()
                                print("Average surface observation error (m):",np.average(np.array(pose_results_dict["surface_observation_error_m"]))) 
                                print("Average position error (m):",np.average(np.array(pose_results_dict["position_offset_m_abs"]))) 
                                print("Average angular error: ",np.average(np.array(pose_results_dict["angular_error"])))
                                print() 
                                print("Median surface observation error (m):",np.median(np.array(pose_results_dict["surface_observation_error_m"]))) 
                                print("Median position error (m):",np.median(np.array(pose_results_dict["position_offset_m_abs"]))) 
                                print("Median angular error: ",np.median(np.array(pose_results_dict["angular_error"])))
                                print()
                                print("Average runtime (s):",np.average(np.array(pose_results_dict["runtime"])))

            f_performance_table.close()

    


#####################################################################################################################
# End of function declarations
#####################################################################################################################



if __name__ == "__main__":
    main()
