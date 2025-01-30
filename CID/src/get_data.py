import cv2
import glob
import math
import numpy as np
import random

from src.Pose import *
from src.Crater import *
from src.utils import *

# Returns a PANGU generated image. 
def get_image(pangu_image_file):
    image = cv2.imread(pangu_image_file)
    return image

# Returns a camera intrinsic matrix.
def get_intrinsic(calibration_file):
    f = open(calibration_file, 'r')
    lines = f.readlines()
    calibration = lines[1].split(',')
    fov = int(calibration[0])
    # fx = int(calibration[1])
    # fy = int(calibration[2])
    image_width = int(calibration[3])
    image_height = int(calibration[4])

    fov = fov*math.pi/180
    fx = image_width/(2*math.tan(fov/2)) # Conversion from fov to focal length
    fy = image_height/(2*math.tan(fov/2)) # Conversion from fov to focal length
    cx = image_width/2
    cy = image_height/2

    return (np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]),image_width,image_height)

# Return the reference and re-oriented pose of the camera from a PANGU flight file.
# The reference (nadir) pose is first, followed by the re-oriented pose.
# Pose structure: (x, y, z, pitch, yaw, roll)
def get_camera_poses(pangu_flight_file):
    f = open(pangu_flight_file, 'r')
    lines = f.readlines()
    lines = [i.split() for i in lines]
    poses = []
    for i in lines:
        # Camera pose line is prefixed with "start" and has structure -> x, y, z, yaw, pitch, roll respectively
        if len(i) > 0 and i[0] == "start":
            pose = np.float_(i[1:])
            poses.append(Pose(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]))
    return poses



@njit
def create_rotation_matrix(axis, angle):
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis

    # Compute sine and cosine of the angle
    c = np.cos(angle)
    s = np.sin(angle)

    # Compute the cross-product matrix K
    K = np.array([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]])

    # Compute the rotation matrix using Rodrigues' formula
    R = np.eye(3) + s * K + (1 - c) * np.dot(K, K)

    return R

@njit
def get_craters_world(lines):
    # Initialize the matrices
    N = len(lines)
    crater_param = np.zeros((N, 6))
    crater_conic = np.zeros((N, 3, 3))
    crater_conic_inv = np.zeros((N, 3, 3))
    Hmi_k = np.zeros((N, 4, 3))

    ENU = np.zeros((N, 3, 3))
    L_prime = np.zeros((N, 3, 3))
    S = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    # Populate the matrices
    k = np.array([0, 0, 1])

    for idx, line in enumerate(lines):
        X, Y, Z, a, b, phi = line
        a = (a * 1000) / 2  # converting diameter to meter
        b = (b * 1000) / 2
        phi = phi

        # Populate crater_param
        crater_param[idx] = [X, Y, Z, a, b, phi]

        # Calculate conic matrix
        A = a ** 2 * (np.sin(phi) ** 2) + b ** 2 * (np.cos(phi) ** 2)
        B = 2 * (b ** 2 - a ** 2) * np.cos(phi) * np.sin(phi)
        C = a ** 2 * (np.cos(phi) ** 2) + b ** 2 * (np.sin(phi) ** 2)
        D = -2 * A * 0 - B * 0
        E = -B * 0 - 2 * C * 0
        F = A * 0 ** 2 + B * 0 * 0 + C * 0 ** 2 - a ** 2 * b ** 2

        # Populate crater_conic
        # crater_conic[idx] = [[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]]
        crater_conic[idx] = np.array([[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]])

        crater_conic_inv[idx] = np.linalg.inv(crater_conic[idx])
        # get ENU coordinate
        Pc_M = np.array([X, Y, Z])

        u = Pc_M / np.linalg.norm(Pc_M)
        e = np.cross(k, u) / np.linalg.norm(np.cross(k, u))
        n = np.cross(u, e) / np.linalg.norm(np.cross(u, e))

        R_on_the_plane = create_rotation_matrix(u, phi)
        e_prime = R_on_the_plane @ e
        n_prime = R_on_the_plane @ n
        curr_L_prime = np.empty((3, 3), dtype=np.float64)
        curr_L_prime[:, 0] = e_prime
        curr_L_prime[:, 1] = n_prime
        curr_L_prime[:, 2] = u

        TE_M = np.empty((3, 3), dtype=np.float64)
        TE_M[:, 0] = e
        TE_M[:, 1] = n
        TE_M[:, 2] = u

        ENU[idx] = TE_M
        # compute Hmi

        Hmi = np.hstack((TE_M.dot(S), Pc_M.reshape(-1, 1)))
        Hmi_k[idx] = np.vstack((Hmi, k.reshape(1, 3)))
        L_prime[idx] = curr_L_prime

    return crater_param, crater_conic, crater_conic_inv, ENU, Hmi_k, L_prime


# Get a list of 3D craters in the world reference frame.
# def get_craters_world(craters_world_file):
#     # Read the file and process the lines
#     with open(craters_world_file, "r") as f:
#         lines = f.readlines()[1:]  # ignore the first line
#     lines = [i.split(',') for i in lines]

#     # Initialize the matrices
#     N = len(lines)
#     crater_param = np.zeros((N, 5))
#     crater_conic = np.zeros((N, 3, 3))
#     crater_conic_inv = np.zeros((N, 3, 3))
#     Hmi_k = np.zeros((N, 4, 3))
#     ENU = np.zeros((N, 3, 3))
#     S = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

#     # Populate the matrices
#     k = np.array([0, 0, 1])
#     for idx, line in enumerate(lines):
#         X, Y, Z, a, b = line
#         X = np.float64(X)
#         Y = np.float64(Y)
#         Z = np.float64(Z)
#         a = np.float64(a)
#         b = np.float64(b)
#         phi = 0  # Assuming circular crater

#         # Populate crater_param
#         crater_param[idx] = [X, Y, Z, a, b]

#         # Calculate conic matrix
#         A = a ** 2 * (np.sin(phi) ** 2) + b ** 2 * (np.cos(phi) ** 2)
#         B = 2 * (b ** 2 - a ** 2) * np.cos(phi) * np.sin(phi)
#         C = a ** 2 * (np.cos(phi) ** 2) + b ** 2 * (np.sin(phi) ** 2)
#         D = -2 * A * 0 - B * 0
#         E = -B * 0 - 2 * C * 0
#         F = A * 0 ** 2 + B * 0 * 0 + C * 0 ** 2 - a ** 2 * b ** 2

#         # Populate crater_conic
#         crater_conic[idx] = [[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]]
#         crater_conic_inv[idx] = np.linalg.inv(crater_conic[idx])

#         # get ENU coordinate
#         Pc_M = np.array([X, Y, Z])

#         u = Pc_M / np.linalg.norm(Pc_M)
#         e = np.cross(k, u) / np.linalg.norm(np.cross(k, u))
#         n = np.cross(u, e) / np.linalg.norm(np.cross(u, e))

#         TE_M = np.array([e, n, u]).T
#         ENU[idx] = TE_M
#         # compute Hmi
#         Hmi = np.hstack((TE_M.dot(S), Pc_M[:, np.newaxis]))

#         Hmi_k[idx] = np.vstack((Hmi, k[np.newaxis, :]))

#     return crater_param, crater_conic, crater_conic_inv, ENU, Hmi_k

def get_craters_cam(craters_cam_file):
    """
    Extract crater parameters and conic matrices from the given file.

    Parameters:
    - craters_cam_file: File containing the parameters of each crater in the camera reference frame.

    Returns:
    - crater_param: [N x 5] matrix containing the parameters of each crater.
    - crater_conic: [N x 3 x 3] matrix containing the conic matrix for each crater.
    """
    with open(craters_cam_file, "r") as f:
        lines = f.readlines()[1:]  # ignore the first line
        lines = [np.float64(i.split(',')) for i in lines]

    N = len(lines)
    crater_param = np.zeros((N, 5))
    crater_conic = np.zeros((N, 3, 3))

    for idx, line in enumerate(lines):
        x, y, a, b, phi = line
        crater_param[idx] = [x, y, a, b, phi]
        crater_conic[idx] = ellipse_to_conic_matrix(x, y, a, b, phi)

    return crater_param, crater_conic

# Get a list of detected craters in the image plane.
# The camera detected craters can be offset if the add_noise flag is on.
def get_craters_cam_old(craters_cam_file, add_noise, mu=0, sigma=5):
    f = open(craters_cam_file,"r")
    lines = f.readlines()[1:] #ignore the first line
    lines = [np.float64(i.split(', ')) for i in lines]
    craters = [Crater_c(i[0], i[1], i[2], i[3], i[4]) for i in lines]
    # if add_noise:
    #     craters = [Crater_c(i[0]+np.random.normal(0,sigma), i[1]+np.random.normal(0,sigma), i[2]+np.random.normal(0,sigma), i[3]+np.random.normal(0,sigma), i[4]+np.random.normal(0,sigma)) for i in lines]
    # else:
    #     craters = [Crater_c(i[0], i[1], i[2], i[3], i[4]) for i in lines]
    return craters

def get_files_in_dir(dir, ext):
    if dir[-1] != "/":
        dir += "/"
    files = glob.glob(dir+"*."+ext)
    files = [file[len(dir):] for file in files]
    return files
