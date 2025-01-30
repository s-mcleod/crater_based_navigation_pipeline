import cv2
from mpmath import mp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Circle, Rectangle, Polygon
from numpy import linspace
import mpl_toolkits.mplot3d.art3d as art3d
from sklearn.cluster import KMeans
import os

#####################################################################################################################
# Start of camera pose visualisations
# MODIFIED: https://github.com/opencv/opencv/pull/10354
#####################################################################################################################

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv

def transform_to_matplotlib_frame(cMo, X, inverse=False):
    # inverse=False
    M = np.identity(4)
    M[1,1] = 0
    M[1,2] = 1
    M[2,1] = -1
    M[2,2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))

def create_camera_model(camera_matrix, width, height, scene_focal, draw_frame_axis=False):
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    # focal = 2 / (fx + fy)
    # f_scale = scale_focal * focal
    f_scale = scene_focal


    # draw image plane
    X_img_plane = np.ones((4,5))
    X_img_plane[0:3,0] = [-width, height, f_scale]
    X_img_plane[0:3,1] = [width, height, f_scale]
    X_img_plane[0:3,2] = [width, -height, f_scale]
    X_img_plane[0:3,3] = [-width, -height, f_scale]
    X_img_plane[0:3,4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4,3))
    X_triangle[0:3,0] = [-width, -height, f_scale]
    X_triangle[0:3,1] = [0, -2*height, f_scale]
    X_triangle[0:3,2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4,2))
    X_center1[0:3,0] = [0, 0, 0]
    X_center1[0:3,1] = [-width, height, f_scale]

    X_center2 = np.ones((4,2))
    X_center2[0:3,0] = [0, 0, 0]
    X_center2[0:3,1] = [width, height, f_scale]

    X_center3 = np.ones((4,2))
    X_center3[0:3,0] = [0, 0, 0]
    X_center3[0:3,1] = [width, -height, f_scale]

    X_center4 = np.ones((4,2))
    X_center4[0:3,0] = [0, 0, 0]
    X_center4[0:3,1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4,2))
    X_frame1[0:3,0] = [0, 0, 0]
    X_frame1[0:3,1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4,2))
    X_frame2[0:3,0] = [0, 0, 0]
    X_frame2[0:3,1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4,2))
    X_frame3[0:3,0] = [0, 0, 0]
    X_frame3[0:3,1] = [0, 0, f_scale/2]
    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]

def draw_camera_boards(ax, camera_matrix, cam_width, cam_height, scene_focal,
                       extrinsics, colour):
    min_values = np.zeros((3,1))
    min_values = np.inf
    max_values = np.zeros((3,1))
    max_values = -np.inf

    X_moving = create_camera_model(camera_matrix, cam_width, cam_height, scene_focal, True)
    cm_subsection = linspace(0.0, 1.0, extrinsics.shape[0])
    colors = [ cm.jet(x) for x in cm_subsection ]

    swap_to = [1,1] #NOTE: not sure why we have to swap these indices 
    swap_from = [1,1] #NOTE: not sure why we have to swap these indices 
    # for idx in range(1):
    for idx in range(extrinsics.shape[0]):
        R, _ = cv2.Rodrigues(extrinsics[idx,0:3])
        cMo = np.eye(4,4)
        cMo[0:3,0:3] = extrinsics[0:3,0:3]
        cMo[0:3,3] = extrinsics[0:3,3]
        # for i in range(1):
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4,j] = transform_to_matplotlib_frame(cMo, X_moving[i][0:4,j], True)
            X[swap_to] = X[swap_from]
            ax.plot3D(X[0,:], X[1,:], X[2,:], color=colour,zorder=1000)
            min_values = np.minimum(min_values, X[0:3,:].min(1))
            max_values = np.maximum(max_values, X[0:3,:].max(1))

    return min_values, max_values

def plot_camera_pose(K, estimated_extrinsic_matrix, ground_truth_extrinsic_matrix, all_craters_world, craters_world, dir, write_dir, pixel_noise, euler_bound, position_bound, img_num, metric, world_camera_surface_intersection_points, interactive = False):
    plt.rcParams["figure.figsize"] = (7,7)
    estimated_extrinsic_matrix = np.dot(np.linalg.inv(K),estimated_extrinsic_matrix)
    estimated_position = np.dot(np.linalg.inv(estimated_extrinsic_matrix[0:3,0:3]),estimated_extrinsic_matrix)[:,3]
    ground_truth_extrinsic_matrix = np.dot(np.linalg.inv(K),ground_truth_extrinsic_matrix)
    ground_truth_position = np.dot(np.linalg.inv(ground_truth_extrinsic_matrix[0:3,0:3]),ground_truth_extrinsic_matrix)[:,3]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal', adjustable='box')

    camera_matrix = K

    # NOTE: probably need to change later
    # Plot for estimated and ground truth (respectively) extrinsics
    extrinsics = []

    extrinsics.append(estimated_extrinsic_matrix)
    extrinsics.append(ground_truth_extrinsic_matrix)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')

    x_lim = [craters_world[0].X, craters_world[0].X]
    y_lim = [craters_world[0].Y, craters_world[0].Y]
    z_lim = [craters_world[0].Z, craters_world[0].Z]


    # Only plot the top 100 craters in the world surface.
    for crater in all_craters_world[:100]:
        x,y,z = crater.get_crater_centre()
        p = Circle((x, -y), crater.a)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z, zdir="y")

        if (crater.X < x_lim[0]):
            x_lim[0] = crater.X
        if (crater.Y < y_lim[0]):
            y_lim[0] = crater.Y
        if (crater.Z < z_lim[0]):
            z_lim[0] = crater.Z

        if (crater.X > x_lim[1]):
            x_lim[1] = crater.X
        if (crater.Y > y_lim[1]):
            y_lim[1] = crater.Y
        if (crater.Z > z_lim[1]):
            z_lim[1] = crater.Z
    # Plot all craters used for pnp.
    # centre_x = [c.X for c in craters_world]
    # centre_y = [-c.Y for c in craters_world]
    # min_x = min(centre_x)
    # min_y = min(centre_y)
    # max_x = max(centre_x)
    # max_y = max(centre_y)
    # p = Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, color="r", fill = False)
    # Plot the camera projection bounds
    points = np.array([[p[0],-1*p[1]] for p in world_camera_surface_intersection_points])
    p = Polygon(np.array(points), closed=True, color="r", fill = False)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="y")
    for crater in craters_world:
        x,y,z = crater.get_crater_centre()
        p = Circle((x, -y), crater.a, color="r")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z, zdir="y")

        if (crater.X < x_lim[0]):
            x_lim[0] = crater.X
        if (crater.Y < y_lim[0]):
            y_lim[0] = crater.Y
        if (crater.Z < z_lim[0]):
            z_lim[0] = crater.Z

        if (crater.X > x_lim[1]):
            x_lim[1] = crater.X
        if (crater.Y > y_lim[1]):
            y_lim[1] = crater.Y
        if (crater.Z > z_lim[1]):
            z_lim[1] = crater.Z
    # Determine an appropriate focal length.
    z_distance = abs(estimated_position)[2]
    scale_factor = 0.4
    # NOTE: Not sure if this is right, this was altitude_w passed as a parameter
    max_altitude = 100000
    scene_focal = z_distance*0.05
    # scene_focal = z_distance*scale_factor
    cam_width = (1024/2)*scale_factor
    cam_height = (1024/2)*scale_factor
    
    # Plot estimated camera
    min_values, max_values = draw_camera_boards(ax, camera_matrix, cam_width, cam_height, scene_focal, extrinsics[0], 'c')
    # Plot ground truth camera
    min_values, max_values = draw_camera_boards(ax, camera_matrix, cam_width, cam_height, scene_focal, extrinsics[1], 'm')

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]

    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5

    if (mid_x - max_range < x_lim[0]):
        x_lim[0] = mid_x - max_range
    if (mid_x + max_range > x_lim[1]):
        x_lim[1] = mid_x + max_range
    if (mid_z - max_range < y_lim[0]):
        y_lim[0] = mid_z - max_range
    if (mid_z + max_range > y_lim[1]):
        y_lim[1] = mid_z + max_range
    if (mid_y - max_range < z_lim[0]):
        z_lim[0] = 0
    if (mid_y + max_range > z_lim[1]):
        z_lim[1] = z_distance

    # For aspect scaling
    if (z_lim[1]-z_lim[0] > 0):
        ax.set_ylim(0, abs(x_lim[0] - x_lim[1]))
    else:
        ax.set_ylim(-1*abs(x_lim[0] - x_lim[1]),0)

    # ax.set_xlim(x_lim[0], x_lim[1])
    # ax.set_ylim(z_lim[0], z_lim[1])
    # ax.set_zlim(y_lim[0], y_lim[1])

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    

    position_diff = np.linalg.norm(estimated_position-ground_truth_position)

    fig.text(0.01, 0.01, 
         "Ground truth position:  %.2f, %.2f, %.2f \nEstimated position:      %.2f, %.2f, %.2f \nPosition difference:      %f"%(ground_truth_position[0],ground_truth_position[1],ground_truth_position[2], estimated_position[0],estimated_position[1],estimated_position[2], position_diff), 
         fontsize = 10,
         color = "black")

    extrinsic_plots_dir = "extrinsic_plots/"+metric+"_metric/"
    if not os.path.isdir(write_dir+extrinsic_plots_dir):
        os.makedirs(write_dir+extrinsic_plots_dir)
    pixel_noise_dir = extrinsic_plots_dir+str(pixel_noise)+"_"+str(int(euler_bound))+"_"+str(int(position_bound))+"/"
    if not os.path.isdir(write_dir+pixel_noise_dir):
        os.makedirs(write_dir+pixel_noise_dir)


    plt.savefig(write_dir+pixel_noise_dir+"extrinsic_plot_"+str(img_num)+"_"+dir[:-1]+".png")
    if (interactive):
        plt.show()
    else:
        plt.close('all')


#####################################################################################################################
# End of camera pose visualisations
#####################################################################################################################