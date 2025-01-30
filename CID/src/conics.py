import math
from mpmath import mp
import numpy as np
import random
import numba
from numba import njit, cuda

# Get a conic matrix from projecting a crater onto an image plane.
# Gives the option to add noise to the projected ellipses by a certain pixel offset.
def normised_and_un_normalised_conic_from_crater(c, un_normalised_c, Pm_c, un_normalised_Pm_c, add_noise = False, noise_offset = 0, max_noise_sigma_pix = 1):
    # Normalised
    k = np.array([0, 0, 1])
    Tl_m = c.get_ENU()#np.eye(3) # define a local coordinate system
    S = np.vstack((np.eye(2), np.array([0,0])))
    Pc_mi = c.get_crater_centre().reshape((3,1)) # get the real 3d crater point in moon coordinates
    Hmi = np.hstack((np.dot(Tl_m,S), Pc_mi))
    Cstar = np.linalg.inv(c.conic_matrix_local)
    Hci  = np.dot(Pm_c, np.vstack((Hmi, np.transpose(k))))
    Astar = np.dot(Hci,np.dot(Cstar, np.transpose(Hci)))
    A = np.linalg.inv(Astar)

    # Un-normalised
    un_normalised_Tl_m = un_normalised_c.get_ENU()#np.eye(3) # define a local coordinate system
    un_normalised_Pc_mi = un_normalised_c.get_crater_centre().reshape((3,1)) # get the real 3d crater point in moon coordinates
    un_normalised_Hmi = np.hstack((np.dot(un_normalised_Tl_m,S), un_normalised_Pc_mi))
    un_normalised_Cstar = np.linalg.inv(un_normalised_c.conic_matrix_local)
    un_normalised_Hci  = np.dot(un_normalised_Pm_c, np.vstack((un_normalised_Hmi, np.transpose(k))))
    un_normalised_Astar = np.dot(un_normalised_Hci,np.dot(un_normalised_Cstar, np.transpose(un_normalised_Hci)))
    un_normalised_A = np.linalg.inv(un_normalised_Astar)


    # Noise is applied to elliptical centre, radii and angle of orientation, sampled from a Gaussian distribution with sigma set 
    # as a proportion of the semi minor axis of the ellipse (pixels). Given sigma is a proportion, we cap sigma at 5 pixels for
    # a "realistic" noise offset for larger craters.
    # TODO: is this the right way to get and apply the scale?
    if (add_noise):
        x_c, y_c, a, b, phi = conic_matrix_to_ellipse(A) # The ellipse parameters will be the same for the normalised and un-normalised conics.
        A_scaled = ellipse_to_conic_matrix(x_c, y_c, a, b, phi)
        s = A[0][0]/A_scaled[0][0]
        un_normalised_s = un_normalised_A[0][0]/A_scaled[0][0]
        # Get pixel offset as a function of the semi minor axis length.
        sigma = min(b * noise_offset, max_noise_sigma_pix)
        x_c += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
        y_c += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
        a += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
        b += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
        phi += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)*(math.pi/180)
        A = s*ellipse_to_conic_matrix(x_c, y_c, a, b, phi)
        un_normalised_A = un_normalised_s*ellipse_to_conic_matrix(x_c, y_c, a, b, phi)

    return A, un_normalised_A

# def conic_from_crater(C_conic_inv, Hmi_k, Pm_c):
#     '''
#     :param Pm_c: [3, 4] projection matrix
#     :param Hmi_k = [Hmi, [0 0 1]] # precomputed during craters reading
#     :param C_conic_inv = [3, 3] # precomputed during craters reading
#     :return:
#     '''
#
#     Hci = Pm_c.dot(Hmi_k)
#     Astar = Hci.dot(C_conic_inv).dot(Hci.T)
#
#     # Compute the inverse of Astar to get A
#     A = np.linalg.inv(Astar)
#     return A
@cuda.jit(device=True)
def matrix_vector_multiply(A, v, result, A_rows, A_cols):
    '''
    result = Matmul(A, v)
    :param A: [A_rows x A_cols]
    :param v: [A_cols]
    :param result: [A_rows]
    :param A_rows: Number of rows in matrix A
    :param A_cols: Number of columns in matrix A (and size of vector v)
    :return:
    '''
    for i in range(A_rows):
        result[i] = 0.0
        for j in range(A_cols):
            result[i] += A[i, j] * v[j]

def matrix_vector_multiply_cpu(A, v, result, A_rows, A_cols):
    '''
    result = Matmul(A, v)
    :param A: [A_rows x A_cols]
    :param v: [A_cols]
    :param result: [A_rows]
    :param A_rows: Number of rows in matrix A
    :param A_cols: Number of columns in matrix A (and size of vector v)
    :return:
    '''
    result = np.zeros_like(v)
    for i in range(A_rows):
        result[i] = 0.0
        for j in range(A_cols):
            result[i] += A[i, j] * v[j]

    return result




@cuda.jit(device=True)
def matrix_multiply_3x3(A, B, C):
    for i in range(3):
        for j in range(3):
            C[i, j] = 0.0
            for k in range(3):
                C[i, j] += A[i, k] * B[k, j]

# @cuda.jit(device=True)
# def matrix_multiply_3x4(A, B, C):
#     '''
#     C = Matmul(A, B)
#     :param A: [3 x 4]
#     :param B: [4 x 3]
#     :param C: [3 x 3]
#     :return:
#     '''
#     for i in range(3):
#         for j in range(4):
#             C[i, j] = 0.0
#             for k in range(3):
#                 C[i, j] += A[i, k] * B[k, j]
@cuda.jit(device=True)
def matrix_multiply(A, B, C, A_rows, A_cols, B_cols):
    '''
    C = Matmul(A, B)
    :param A: [A_rows x A_cols]
    :param B: [A_cols x B_cols]
    :param C: [A_rows x B_cols]
    :param A_rows: Number of rows in matrix A
    :param A_cols: Number of columns in matrix A (and number of rows in matrix B)
    :param B_cols: Number of columns in matrix B
    :return:
    '''
    for i in range(A_rows):
        for j in range(B_cols):
            C[i, j] = 0.0
            for k in range(A_cols):
                C[i, j] += A[i, k] * B[k, j]

def matrix_multiply_cpu(A, B, A_rows, A_cols, B_cols):
    C = np.zeros([A_rows, B_cols])
    for i in range(A_rows):
        for j in range(B_cols):
            C[i, j] = 0.0
            for k in range(A_cols):
                C[i, j] += A[i, k] * B[k, j]

    return C



@cuda.jit(device=True)
def inverse_3x3(A, invA):
    detA = A[0, 0] * (A[1, 1] * A[2, 2] - A[2, 1] * A[1, 2]) - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
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

def inverse_3x3_cpu(A):
    detA = A[0, 0] * (A[1, 1] * A[2, 2] - A[2, 1] * A[1, 2]) - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) + \
           A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    invDetA = 1.0 / detA
    invA = np.zeros_like(A)

    invA[0, 0] = (A[1, 1] * A[2, 2] - A[2, 1] * A[1, 2]) * invDetA
    invA[0, 1] = (A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]) * invDetA
    invA[0, 2] = (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]) * invDetA
    invA[1, 0] = (A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]) * invDetA
    invA[1, 1] = (A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]) * invDetA
    invA[1, 2] = (A[1, 0] * A[0, 2] - A[0, 0] * A[1, 2]) * invDetA
    invA[2, 0] = (A[1, 0] * A[2, 1] - A[2, 0] * A[1, 1]) * invDetA
    invA[2, 1] = (A[2, 0] * A[0, 1] - A[0, 0] * A[2, 1]) * invDetA
    invA[2, 2] = (A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]) * invDetA
    return invA


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
    A = inverse_3x3_cpu(Astar)
    return A


@cuda.jit(device=True)
def conic_from_crater(C_conic_inv, Hmi_k, Pm_c, A):
    '''

    :param C_conic_inv: [3x3]
    :param Hmi_k: [4x3]
    :param Pm_c: [3x4]
    :param A: [3x3]
    :return:
    '''
    Hci = cuda.local.array((3, 4), dtype=numba.float64)
    Astar_tmp = cuda.local.array((3, 3), dtype=numba.float64)
    Astar = cuda.local.array((3, 3), dtype=numba.float64)

    # matrix_multiply_3x4(Pm_c, Hmi_k, Hci)
    matrix_multiply(Pm_c, Hmi_k, Hci, 3, 4, 3)
    matrix_multiply(Hci, C_conic_inv, Astar_tmp, 3, 3, 3)
    matrix_multiply(Astar_tmp, Hci.T, Astar, 3, 3, 3)
    inverse_3x3(Astar, A)

# @jit(nopython=True, nogil=True, cache=True)
@njit
def conic_from_crater_opt(c, Pm_c, add_noise=False, noise_offset=0, max_noise_sigma_pix=1):
    k = np.array([0, 0, 1]).reshape((3, 1))

    # Use np.eye directly without vstack
    S = np.array([[1, 0], [0, 1], [0, 0]])
    S = S.astype(np.float64)

    Pc_mi = c.get_crater_centre().reshape((3, 1))
    Pc_mi = Pc_mi.astype(np.float64)

    Hmi = np.hstack((c.get_ENU().dot(S), Pc_mi))

    Cstar_inv = np.linalg.inv(c.conic_matrix_local)

    # Combine dot products
    Hci = Pm_c.dot(np.vstack((Hmi, k.T)))
    Astar = Hci.dot(Cstar_inv).dot(Hci.T)

    # Compute the inverse of Astar to get A
    A = np.linalg.inv(Astar)

    return A

# Get a conic matrix from projecting a crater onto an image plane.
# Gives the option to add noise to the projected ellipses by a certain pixel offset.
def conic_from_crater_old(c, Pm_c, add_noise = False, noise_offset = 0, max_noise_sigma_pix = 1):
    k = np.array([0, 0, 1])
    Tl_m = c.get_ENU()#np.eye(3) # define a local coordinate system
    S = np.vstack((np.eye(2), np.array([0,0])))
    Pc_mi = c.get_crater_centre().reshape((3,1)) # get the real 3d crater point in moon coordinates
    Hmi = np.hstack((np.dot(Tl_m,S), Pc_mi))
    Cstar = np.linalg.inv(c.conic_matrix_local)

    Hci  = np.dot(Pm_c, np.vstack((Hmi, np.transpose(k))))
    Astar = np.dot(Hci,np.dot(Cstar, np.transpose(Hci)))
    A = np.linalg.inv(Astar)


    # Noise is applied to elliptical centre, radii and angle of orientation, sampled from a Gaussian distribution with sigma set 
    # as a proportion of the semi minor axis of the ellipse (pixels). Given sigma is a proportion, we cap sigma at 5 pixels for
    # a "realistic" noise offset for larger craters.
    # TODO: is this the right way to get and apply the scale?
    if (add_noise):
        x_c, y_c, a, b, phi = conic_matrix_to_ellipse(A)
        A_scaled = ellipse_to_conic_matrix(x_c, y_c, a, b, phi)
        # TODO: is this correct?
        s = A[0][0]/A_scaled[0][0]
        # Get pixel offset as a function of the semi minor axis length.
        sigma = min(b * noise_offset, max_noise_sigma_pix)
        sigma = b*noise_offset #TODO: REMOVE # noise_offset
        # if (sigma == max_noise_sigma_pix):
        #     print("maxed out")
        x_c +=  [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
        y_c +=  [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
        a += 0#[-1,1][random.randrange(2)]*np.random.normal(0, sigma)
        b +=  0#[-1,1][random.randrange(2)]*np.random.normal(0, sigma)
        phi += 0#[-1,1][random.randrange(2)]*np.random.normal(0, sigma)*(math.pi/180)
        A = s*ellipse_to_conic_matrix(x_c, y_c, a, b, phi)
    
    # TODO: is it correct to normalise?
    A = A #/np.linalg.norm(A)
    return A

# Get elliptical parameters from a conic matrix.
def conic_matrix_to_ellipse(cm):
    A = cm[0][0]
    B = cm[0][1]*2
    C = cm[1][1]
    D = cm[0][2]*2
    E = cm[1][2]*2
    F = cm[2][2]

    x_c = (2*C*D-B*E)/(B**2-4*A*C)
    y_c = (2*A*E-B*D)/(B**2-4*A*C)

    if ((B**2-4*A*C) >= 0):
        return 0,0,0,0,0

    try:
        a = math.sqrt((2*(A*E**2+C*D**2 - B*D*E + F*(B**2-4*A*C)))/((B**2-4*A*C)*(math.sqrt((A-C)**2+B**2)-A-C)))
        b = math.sqrt((2*(A*E**2+C*D**2 - B*D*E + F*(B**2-4*A*C)))/((B**2-4*A*C)*(-1*math.sqrt((A-C)**2+B**2)-A-C)))

        phi = 0
        if (B == 0 and A > C):
            phi = math.pi/2
        elif (B != 0 and A <= C):
            phi = 0.5*mp.acot((A-C)/B)
        elif (B != 0 and A > C):
            phi = math.pi/2+0.5*mp.acot((A-C)/B)
        
        return x_c, y_c, a, b, phi
    
    except:
        return 0,0,0,0,0

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
