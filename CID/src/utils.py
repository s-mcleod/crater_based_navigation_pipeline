import numpy as np
import math
import numba
from numba import njit
import os
os.environ['OPENCV_IO_ENABLE_JASPER']='true'
import cv2
import os
import matplotlib.pyplot as plt
# from src.metrics_ck import *

from mpmath import mp


@njit
def custom_meshgrid(x, y, z):  # tested
    nx, ny, nz = len(x), len(y), len(z)

    x_grid = np.empty((ny, nx, nz))
    y_grid = np.empty((ny, nx, nz))
    z_grid = np.empty((ny, nx, nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x_grid[j, i, k] = x[i]
                y_grid[j, i, k] = y[j]
                z_grid[j, i, k] = z[k]

    return x_grid, y_grid, z_grid

def custom_meshgrid_2d(x, y):  # tested
    nx, ny= len(x), len(y)

    x_grid = np.empty((ny, nx))
    y_grid = np.empty((ny, nx))

    for i in range(nx):
        for j in range(ny):
            x_grid[j, i] = x[i]
            y_grid[j, i] = y[j]


    return x_grid, y_grid





def add_noise_to_craters_cam(craters_cam, a_noise, b_noise, x_noise, y_noise, phi_noise):
    noisy_craters_cam = np.zeros([craters_cam.shape[0], 3, 3])
    noisy_craters_params = np.zeros([craters_cam.shape[0], 5])
    a_noise_pct = []
    b_noise_pct = []
    x_noise_pct = []
    y_noise_pct = []
    for i in range(len(craters_cam)):
        curr_x = craters_cam[i, 0] + x_noise[i]
        curr_y = craters_cam[i, 1] + y_noise[i]
        curr_a = craters_cam[i, 2] + a_noise[i]
        curr_b = craters_cam[i, 3] + b_noise[i]
        curr_phi = craters_cam[i, 4] + phi_noise[i]
        x_noise_pct.append(x_noise[i] / craters_cam[i, 0])
        y_noise_pct.append(y_noise[i] / craters_cam[i, 1])
        a_noise_pct.append(a_noise[i] / craters_cam[i, 2])
        b_noise_pct.append(b_noise[i] / craters_cam[i, 3])

        noisy_craters_params[i] = curr_x, curr_y, curr_a, curr_b, curr_phi
        noisy_craters_cam[i] = ellipse_to_conic_matrix(curr_x, curr_y, curr_a, curr_b, curr_phi)

    return noisy_craters_cam, noisy_craters_params, x_noise_pct, y_noise_pct, a_noise_pct, b_noise_pct

def craters_weight_computation(craters_params):
    size = []
    for i in range(len(craters_params)):
        curr_a = craters_params[i, 2]
        curr_b = craters_params[i, 3]
        size.append(curr_a * curr_b)

    sum = np.sum(np.array(size))
    weights = np.array(size) / sum

    return weights


def generate_oriented_bbox_points_cpu(CC_params, num_sam):
    N = CC_params.shape[0]
    MAX_SAMPLES = num_sam ** 2  # Set this to the maximum number of samples you expect

    rotated_points_x = np.zeros((N, MAX_SAMPLES))
    rotated_points_y = np.zeros((N, MAX_SAMPLES))
    level_curve_a = np.zeros((N, MAX_SAMPLES))

    for k in range(N):
        xc, yc, a, b, phi = CC_params[k]

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
                rotated_points_x[k, idx] = rotated_point[0] + xc
                rotated_points_y[k, idx] = rotated_point[1] + yc

                disp_a = np.array([rotated_points_x[k, idx], rotated_points_y[k, idx]]) - np.array([xc, yc])
                level_curve_a[k, idx] = np.transpose(disp_a) @ D_a @ disp_a
                idx += 1

    return rotated_points_x, rotated_points_y, level_curve_a

# # Example usage:
# CC_params = np.array([[0, 0, 1, 1, 0], [1, 1, 2, 2, math.pi / 4]])
# rotated_points_x, rotated_points_y = generate_oriented_bbox_points_cpu(CC_params, 10)



def round_up(value, decimals=2):
    return math.floor(value * 10**decimals) / 10**decimals

def angular_distance(R1, R2):
    """
    Compute the angular distance between two rotation matrices R1 and R2.

    Parameters:
    - R1, R2: Rotation matrices.

    Returns:
    - Angular distance in radians.
    """
    # Compute the relative rotation matrix
    # R = np.dot(R2, R1.T)
    R = np.dot(R1.T, R2)
    # Compute the angle of rotation
    theta = np.arccos((np.trace(R) - 1) / 2.0)

    return np.rad2deg(theta)



# Get a conic matrix from an ellipse.
@njit
def ellipse_to_conic_matrix(x, y, a, b, phi):
    A = a**2*((np.sin(phi))**2)+b**2*((np.cos(phi))**2)
    B = 2*(b**2-a**2)*np.cos(phi)*np.sin(phi)
    C = a**2*((np.cos(phi))**2)+b**2*((np.sin(phi))**2)
    D = -2*A*x-B*y
    E = -B*x-2*C*y
    F = A*x**2+B*x*y+C*y**2-a**2*b**2

    # TODO: do i need to normalise here?

    return np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]])



def create_extrinsic_matrix(plane_normal, radius, rotate=False):
    # Ensure the plane normal is a unit vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Camera's z-axis is the opposite of the plane normal
    z_axis = -plane_normal

    # Determine an up vector. If the z-axis is not parallel to [0, 1, 0], use [0, 1, 0] as the up vector.
    # Otherwise, use [1, 0, 0].
    if np.abs(np.dot(z_axis, [0, 1, 0])) != 1:
        up_vector = [0, 1, 0]
    else:
        up_vector = [1, 0, 0]

    # Camera's x-axis
    x_axis = np.cross(up_vector, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Camera's y-axis
    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrix
    R = np.array([x_axis, y_axis, z_axis]).T

    rotation_angle = 0
    if rotate:
        # Compute a random rotation angle between 0 and 60 degrees
        rotation_angle = np.random.uniform(0, np.radians(60))

        # Create a rotation matrix around a random axis
        random_axis = np.random.rand(3)
        random_axis = random_axis / np.linalg.norm(random_axis)

        rand_rot_mat = axis_angle_to_rotation_matrix_scipy(random_axis, rotation_angle)
        # Apply the random rotation to R
        R = R @ rand_rot_mat

    # Translation vector (camera's position in world coordinates)
    t = plane_normal * radius

    # Extrinsic matrix
    extrinsic = np.zeros((3, 4))
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = -R.T @ t  # Convert world position to camera-centric position
    # extrinsic[3, 3] = 1

    return extrinsic, np.degrees(rotation_angle)

from scipy.spatial.transform import Rotation
def axis_angle_to_rotation_matrix_scipy(axis, angle):
    """
    Convert axis-angle to rotation matrix using scipy.

    Parameters:
    - axis: A 3D unit vector representing the rotation axis.
    - angle: Rotation angle in radians.

    Returns:
    - 3x3 rotation matrix.
    """
    r = Rotation.from_rotvec(axis * angle)
    return r.as_matrix()

def conic_matrix_to_ellipse(cm):
    A = cm[0][0]
    B = cm[0][1] * 2
    C = cm[1][1]
    D = cm[0][2] * 2
    E = cm[1][2] * 2
    F = cm[2][2]

    x_c = (2 * C * D - B * E) / (B ** 2 - 4 * A * C)
    y_c = (2 * A * E - B * D) / (B ** 2 - 4 * A * C)

    if ((B ** 2 - 4 * A * C) >= 0):
        return 0, 0, 0, 0, 0

    try:
        a = math.sqrt((2 * (A * E ** 2 + C * D ** 2 - B * D * E + F * (B ** 2 - 4 * A * C))) / (
                    (B ** 2 - 4 * A * C) * (math.sqrt((A - C) ** 2 + B ** 2) - A - C)))
        b = math.sqrt((2 * (A * E ** 2 + C * D ** 2 - B * D * E + F * (B ** 2 - 4 * A * C))) / (
                    (B ** 2 - 4 * A * C) * (-1 * math.sqrt((A - C) ** 2 + B ** 2) - A - C)))

        phi = 0
        if (B == 0 and A > C):
            phi = math.pi / 2
        elif (B != 0 and A <= C):
            phi = 0.5 * math.acot((A - C) / B)
        elif (B != 0 and A > C):
            phi = math.pi / 2 + 0.5 * math.acot((A - C) / B)

        return x_c, y_c, a, b, phi

    except:
        return 0, 0, 0, 0, 0

@njit
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

        # phi = 0
        # if (B == 0 and A > C):
        #     phi = math.pi/2
        # elif (B != 0 and A <= C):
        #     phi = 0.5*mp.acot((A-C)/B)
        # elif (B != 0 and A > C):
        #     phi = math.pi/2+0.5*mp.acot((A-C)/B)
        
        # # Assuming this will be converted to an int for pixels, if a == b, then phi should be 0.
        # if (abs(a-b) < 0.01):
        #     phi = 0

        if B != 0:
            phi = math.atan((C - A - root) / B)  # Wikipedia had this as acot; should be atan. Check https://math.stackexchange.com/questions/1839510/how-to-get-the-correct-angle-of-the-ellipse-after-approximation/1840050#1840050
        elif A < C:
            phi = 0
        else:
            phi = math.pi / 2

        return True, x_c, y_c, a, b, phi
    except:
        return False, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001

@njit
def conic_from_crater_cpu(C_conic_inv, Hmi_k, Pm_c):
    '''
    :param C_conic_inv: [3x3]
    :param Hmi_k: [4x3]
    :param Pm_c: [3x4]
    :param A: [3x3]
    :return:
    '''
    # Hci = np.dot(Pm_c, Hmi_k)
    Hci = matrix_multiply_cpu(Pm_c, Hmi_k, 3, 4, 3)
    # Astar = np.dot(np.dot(Hci, C_conic_inv), Hci.T)
    Astar = matrix_multiply_cpu(Hci, C_conic_inv, 3, 3, 3)
    Astar = matrix_multiply_cpu(Astar, Hci.T, 3, 3, 3)
    # A = np.linalg.inv(Astar)
    legit_flag, A = inverse_3x3_cpu(Astar)

    return legit_flag, A


def differentiate_values(vec):
    # Compute pairwise absolute differences
    diffs = np.abs([
        vec[0] - vec[1],
        vec[0] - vec[2],
        vec[1] - vec[2]
    ])

    # Find the indices of the two smallest differences
    sorted_indices = np.argsort(diffs)

    # Use the indices to determine the repeated and unique values
    if sorted_indices[0] == 0:  # vec[0] and vec[1] are close
        same_value = np.mean([vec[0], vec[1]])
        unique_value = vec[2]
        unique_idx = 2
    elif sorted_indices[0] == 1:  # vec[0] and vec[2] are close
        same_value = np.mean([vec[0], vec[2]])
        unique_value = vec[1]
        unique_idx = 1
    else:  # vec[1] and vec[2] are close
        same_value = np.mean([vec[1], vec[2]])
        unique_value = vec[0]
        unique_idx = 0

    return same_value, unique_value, unique_idx


@njit
def differentiate_values_numba(vec):
    # Compute pairwise absolute differences using statically-sized arrays
    diffs = np.empty(3)
    diffs[0] = np.abs(vec[0] - vec[1])
    diffs[1] = np.abs(vec[0] - vec[2])
    diffs[2] = np.abs(vec[1] - vec[2])

    # Find the indices of the two smallest differences
    sorted_indices = np.argsort(diffs)

    # Use the indices to determine the repeated and unique values
    if sorted_indices[0] == 0:  # vec[0] and vec[1] are close
        same_value = (vec[0] + vec[1]) / 2
        unique_value = vec[2]
        unique_idx = 2
    elif sorted_indices[0] == 1:  # vec[0] and vec[2] are close
        same_value = (vec[0] + vec[2]) / 2
        unique_value = vec[1]
        unique_idx = 1
    else:  # vec[1] and vec[2] are close
        same_value = (vec[1] + vec[2]) / 2
        unique_value = vec[0]
        unique_idx = 0

    return same_value, unique_value, unique_idx

@njit
def conic_from_crater_cpu_mod(C_conic, Hmi_k, Pm_c):
    '''
    :param C_conic_inv: [3x3]
    :param Hmi_k: [4x3]
    :param Pm_c: [3x4]
    :param A: [3x3]
    :return:
    '''
    # Hci = np.dot(Pm_c, Hmi_k)
    Hci = matrix_multiply_cpu(Pm_c, Hmi_k, 3, 4, 3)
    Hci_inv = np.linalg.inv(Hci)
    # Astar = np.dot(np.dot(Hci, C_conic_inv), Hci.T)
    Astar = matrix_multiply_cpu(Hci_inv.T, C_conic, 3, 3, 3)
    A = matrix_multiply_cpu(Astar, Hci_inv, 3, 3, 3)

    # A_ = Hci.T @ A @ Hci
    # A = np.linalg.inv(Astar)
    # A = inverse_3x3_cpu(Astar)
    return A

def imaged_conic_to_crater_conic(C_conic, Hmi_k, Pm_c):
    '''
    :param C_conic_inv: [3x3]
    :param Hmi_k: [4x3]
    :param Pm_c: [3x4]
    :param A: [3x3]
    :return:
    '''
    # Hci = np.dot(Pm_c, Hmi_k)
    Hci = matrix_multiply_cpu(Pm_c, Hmi_k, 3, 4, 3)
    # Hci_inv = np.linalg.inv(Hci)
    # Astar = np.dot(np.dot(Hci, C_conic_inv), Hci.T)
    Astar = matrix_multiply_cpu(Hci.T, C_conic, 3, 3, 3)
    A = matrix_multiply_cpu(Astar, Hci, 3, 3, 3)
    # A = np.linalg.inv(Astar)
    # A = inverse_3x3_cpu(Astar)
    return A


@njit
def rotation_matrix_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])



@njit
def rotation_matrix_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@njit
def rotation_matrix_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

@njit
def intrinsic_zxz_rotation(alpha, beta, gamma):
    zx_rot = rotation_matrix_x(beta) @ (rotation_matrix_z(gamma))
    return rotation_matrix_z(alpha) @ (zx_rot)



@njit
def extrinsic_xyz_rotation(alpha, beta, gamma):
    return rotation_matrix_x(alpha) @ (rotation_matrix_y(beta) @ rotation_matrix_z(gamma))


@njit
def rotation_compute(yaw_rad, pitch_rad, roll_rad):
    R_w_ci_intrinsic = intrinsic_zxz_rotation(0.0, -np.pi / 2, 0.0)
    R_ci_cf_intrinsic = intrinsic_zxz_rotation(yaw_rad, pitch_rad, 0.0)
    R_c_intrinsic = R_ci_cf_intrinsic @ R_w_ci_intrinsic
    R_w_c_extrinsic = np.transpose(R_c_intrinsic)
    R_c_roll_extrinsic = extrinsic_xyz_rotation(0.0, 0.0, roll_rad)
    R = R_c_roll_extrinsic @ R_w_c_extrinsic
    return R

@njit
def matrix_multiply_cpu(A, B, A_rows, A_cols, B_cols):
    C = np.zeros((A_rows, B_cols))
    for i in range(A_rows):
        for j in range(B_cols):
            C[i, j] = 0.0
            for k in range(A_cols):
                C[i, j] += A[i, k] * B[k, j]

    return C

@njit
def inverse_3x3_cpu(A):

    detA = A[0, 0] * (A[1, 1] * A[2, 2] - A[2, 1] * A[1, 2]) - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) + \
           A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])

    invA = np.zeros_like(A)

    try:
        invDetA = 1.0 / detA
        invA[0, 0] = (A[1, 1] * A[2, 2] - A[2, 1] * A[1, 2]) * invDetA
        invA[0, 1] = (A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]) * invDetA
        invA[0, 2] = (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]) * invDetA
        invA[1, 0] = (A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]) * invDetA
        invA[1, 1] = (A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]) * invDetA
        invA[1, 2] = (A[1, 0] * A[0, 2] - A[0, 0] * A[1, 2]) * invDetA
        invA[2, 0] = (A[1, 0] * A[2, 1] - A[2, 0] * A[1, 1]) * invDetA
        invA[2, 1] = (A[2, 0] * A[0, 1] - A[0, 0] * A[2, 1]) * invDetA
        invA[2, 2] = (A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]) * invDetA
        legit_flag = True
    except:
        legit_flag = False
    return legit_flag, invA

