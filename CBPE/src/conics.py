import math
from mpmath import mp
import numpy as np
import random
import itertools
from numpy.linalg import eig
from itertools import product
from src.Crater import *

from scipy.spatial.transform import Rotation as R

# Get a conic matrix from projecting a crater onto an image plane.
# Gives the option to add noise to the projected ellipses by a certain pixel offset.
def normalised_and_un_normalised_conic_from_crater(c, un_normalised_c, crater_cam, Pm_c, un_normalised_Pm_c, add_noise = False, noise_offset = 0, max_noise_sigma_pix = 1, continuous = False):
    # Normalised
    A = conic_from_crater(c, Pm_c)
    # Un-normalised
    un_normalised_A = conic_from_crater(un_normalised_c, un_normalised_Pm_c)

    # Generally, we want to obtain the craters from their detected image coordinates, but other times we want their continuous true values (for testing).
    if (not continuous):
        x_c, y_c, a, b, phi = crater_cam.x, crater_cam.y, crater_cam.a, crater_cam.b, crater_cam.phi
        A = ellipse_to_conic_matrix(x_c, y_c, a, b, phi)
        un_normalised_A = ellipse_to_conic_matrix(x_c, y_c, a, b, phi)

        # TODO: is this the right way to get and apply the scale?
        if (add_noise):
            x_c, y_c, a, b, phi = crater_cam.x,crater_cam.y, crater_cam.a, crater_cam.b, crater_cam.phi # conic_matrix_to_ellipse(A) # The ellipse parameters will be the same for the normalised and un-normalised conics.
            

            # Random noise
            if noise_offset == 0.1:
                sigma = min(b * noise_offset, 1)
                x_c += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
                y_c += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
                a += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
                b += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
                phi += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)*(math.pi/180)

            # Random noise
            elif noise_offset == 0.2:
                sigma = min(b * noise_offset, 2)
                x_c += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
                y_c += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
                a += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
                b += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
                phi += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)*(math.pi/180)

            elif noise_offset == 0.21:
                x_c += np.random.normal(0.2, max_noise_sigma_pix,2)[0]
                y_c += np.random.normal(0.2, max_noise_sigma_pix,2)[0]
                a += np.random.normal(-0.2, max_noise_sigma_pix,2)[0]
                b += np.random.normal(-0.2, max_noise_sigma_pix,2)[0]




            # # A_scaled = ellss = un_normalised_A[0][0]/A_scaled[0][0]
            # # Get pixel offset as a function of the semi minor axis length.
            # # if noise_offset == 0.1:
            # #     sigma = min(b * noise_offset, max_noise_sigma_pix)
            # #     x_c += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
            # #     y_c += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
            # #     a += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
            # #     b += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)
            # #     phi += [-1,1][random.randrange(2)]*np.random.normal(0, sigma)*(math.pi/180)

            # if noise_offset == 0.1: 
            #     x_c += np.random.normal(-0.20, 1.578,1)[0]
            #     y_c += np.random.normal(0.48, 1.27,1)[0]
            #     a += np.random.normal(-0.12, 2.33,1)[0]
            #     b += np.random.normal(-0.66, 2.10,1)[0]
            
            # elif noise_offset == 0.01:
            #     x_c += np.random.normal(-0.020, .1578,1)[0]
            #     y_c += np.random.normal(0.048, .127,1)[0]
            #     a += np.random.normal(-0.012, .233,1)[0]
            #     b += np.random.normal(-0.066, .210,1)[0]

            # elif noise_offset == 1: 
            #     x_c += np.random.normal(-0.4, 1.7,1)[0]
            #     y_c += np.random.normal(0.6, 1.4,1)[0]
            #     a += np.random.normal(-0.4, 3,1)[0]
            #     b += np.random.normal(-1, 2.5,1)[0]


            # # if noise_offset == 0.2:
            # #     x_c += np.random.normal(-0.1, 1, 1)[0]
            # #     y_c += np.random.normal(0.1, 1, 1)[0]
            # #     a += np.random.normal(-0.1, 1, 1)[0]
            # #     b += np.random.normal(-0.1, 1, 1)[0]
            
            # # elif noise_offset == 0.3:
            # #     x_c += np.random.normal(0.1, 1, 1)[0]
            # #     y_c += np.random.normal(-0.1, 1, 1)[0]
            # #     a += np.random.normal(-0.1, 1, 1)[0]
            # #     b += np.random.normal(-0.1, 1, 1)[0]

            # # # elif noise_offset == .3:
            # # #     x_c += np.random.normal(-0.2, 2, 2)[0]
            # # #     y_c += np.random.normal(0.2, 2, 2)[0]
            # # #     a += np.random.normal(-0.2, 2, 2)[0]
            # # #     b += np.random.normal(-0.2, 2, 2)[0]

            # # elif noise_offset == .4:
            # #     x_c += np.random.normal(-0.5, 2, 1)[0]
            # #     y_c += np.random.normal(0.5, 2, 1)[0]
            # #     a += np.random.normal(-0.4, 3, 1)[0]
            # #     b += np.random.normal(-0.6, 3, 1)[0]
            
            # # elif noise_offset == 0.5:
            # #     #original
            # #     x_c += np.random.normal(-0.20411007054343705, 1.5799928871877509,1)[0]
            # #     y_c += np.random.normal(0.47607361948358995, 1.2748540558318047,1)[0]
            # #     a += np.random.normal(-0.1206451312108643, 2.328082714942255,1)[0]
            # #     b += np.random.normal(-0.6568906500398782, 2.096907100320232,1)[0]
            
            # # elif noise_offset == .6:
            # #     x_c += np.random.normal(0.20411007054343705, 1.5799928871877509,1)[0]
            # #     y_c += np.random.normal(-0.47607361948358995, 1.2748540558318047,1)[0]
            # #     a += np.random.normal(-0.1206451312108643, 2.328082714942255,1)[0]
            # #     b += np.random.normal(-0.6568906500398782, 2.096907100320232,1)[0]

            # # elif noise_offset == .7:
            # #     x_c += np.random.normal(0.20411007054343705, 1.5799928871877509,1)[0]
            # #     y_c += np.random.normal(-0.47607361948358995, 1.2748540558318047,1)[0]
            # #     a += np.random.normal(0.1206451312108643, 2.328082714942255,1)[0]
            # #     b += np.random.normal(0.6568906500398782, 2.096907100320232,1)[0]

            # A = s*ellipse_to_conic_matrix(x_c, y_c, a, b, phi)
            # un_normalised_A = un_normalised_s*ellipse_to_conic_matrix(x_c, y_c, a, b, phi)
            A = ellipse_to_conic_matrix(x_c, y_c, a, b, phi)
            un_normalised_A = ellipse_to_conic_matrix(x_c, y_c, a, b, phi)



    return A, un_normalised_A


# Get a conic matrix from projecting a crater onto an image plane.
# Gives the option to add noise to the projected ellipses by a certain pixel offset.
def conic_from_crater(c, Pm_c, add_noise = False, noise_offset = 0, max_noise_sigma_pix = 1):

    k = np.array([0, 0, 1])
    Tl_m = c.get_ENU()#np.eye(3) # define a local coordinate system
    S = np.vstack((np.eye(2), np.array([0,0])))
    Pc_mi = c.get_crater_centre().reshape((3,1)) # get the real 3d crater point in moon coordinates
    Hmi = np.hstack((np.dot(Tl_m,S), Pc_mi))
    Cstar = np.linalg.inv(c.conic_matrix_local)

    Hci  = np.dot(Pm_c, np.vstack((Hmi, np.transpose(k))))
    Astar = np.dot(Hci,np.dot(Cstar, np.transpose(Hci)))
    A = np.linalg.inv(Astar)
    
    # TODO: is it correct to normalise?
    A = A #/np.linalg.norm(A)
    return A


# Get a conic matrix from projecting a crater onto an image plane.
# Gives the option to add noise to the projected ellipses by a certain pixel offset.
def conic_from_crater2(c, Pm_c, add_noise = False, noise_offset = 0, max_noise_sigma_pix = 1):
    k = np.array([0, 0, 1])

    u = np.array([0,math.pi/5,-math.pi/5])/np.linalg.norm(np.array([0,math.pi/5,-math.pi/5]))
    e = np.cross(k, u)/np.linalg.norm(np.cross(k, u))
    n = np.cross(u, e)/np.linalg.norm(np.cross(u, e))
    Tl_m = np.transpose(np.array([e, n, u]))
    # Tl_m = -np.eye(3)

    S = np.vstack((np.eye(2), np.array([0,0])))
    Pc_mi = c.get_crater_centre().reshape((3,1)) # get the real 3d crater point in moon coordinates
    Hmi = np.hstack((np.dot(Tl_m,S), Pc_mi))
    Cstar = np.linalg.inv(c.conic_matrix_local)

    Hci  = np.dot(Pm_c, np.vstack((Hmi, np.transpose(k))))
    Astar = np.dot(Hci,np.dot(Cstar, np.transpose(Hci)))
    A = np.linalg.inv(Astar)

    return A

def extract_characteristic_points(C):
    x, y, a, b, phi = conic_matrix_to_ellipse(C)

    p00 = x
    p01 = y

    p10 = x + a*math.cos(phi)
    p11 = y + a*math.sin(phi)

    p20 = x - b*math.sin(phi)
    p21 = y + b*math.cos(phi)

    p30 = x - a*math.cos(phi)
    p31 = y - a*math.sin(phi)

    p40 = x + b*math.sin(phi)
    p41 = y - b*math.cos(phi)

    return np.array([p00, p01, p10, p11, p20, p21, p30, p31, p40, p41])

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
        
        # Assuming this will be converted to an int for pixels, if a == b, then phi should be 0.
        if (abs(a-b) < 0.01):
            phi = 0
        return x_c, y_c, a, b, float(phi)
    
    except:
        return 0,0,0,0,0

# Get a conic matrix from an ellipse.
def ellipse_to_conic_matrix(x, y, a, b, phi):
    A = a**2*((math.sin(phi))**2)+b**2*((math.cos(phi))**2)
    B = 2*(b**2-a**2)*math.cos(phi)*math.sin(phi)
    C = a**2*((math.cos(phi))**2)+b**2*((math.sin(phi))**2)
    D = -2*A*x-B*y
    E = -B*x-2*C*y
    F = A*x**2+B*x*y+C*y**2-a**2*b**2

    # TODO: do i need to normalise here?

    return np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]])

# Find the indices of the three image points that are closest together.
def closest_three_points_indices(craters_world):
    points = []
    for c in craters_world:
        points.append((c.X, c.Y, c.Z))
    
    # Function to calculate Euclidean distance between two points
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
    
    # Generate all possible triplets of point indices
    triplet_indices = itertools.combinations(range(len(points)), 3)
    
    # Find the triplet with the minimum total distance (sum of pairwise distances)
    min_triplet = None
    min_distance_sum = float('inf')
    
    for triplet in triplet_indices:
        # Get the actual points for the current triplet of indices
        p1, p2, p3 = points[triplet[0]], points[triplet[1]], points[triplet[2]]
        
        # Calculate the sum of pairwise distances in the triplet
        dist_sum = (euclidean_distance(p1, p2) +
                    euclidean_distance(p2, p3) +
                    euclidean_distance(p1, p3))
        
        # Update the minimum triplet if necessary
        if dist_sum < min_distance_sum:
            min_triplet = triplet
            min_distance_sum = dist_sum
    
    return list(min_triplet)

def angle_between_vectors(v1, v2):
    # Convert inputs to numpy arrays for easier vector operations
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Compute the dot product of the two vectors
    dot_product = np.dot(v1, v2)
    
    # Compute the magnitudes (norms) of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Compute the cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Ensure the value is within the valid range for acos due to potential floating-point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Compute the angle in radians
    angle = np.arccos(cos_theta)
    
    return angle

# Find the norms of the three closest craters whose norms are maximally aligned.
def max_alignment(dual_craters, closest_points_indices, true_duals_indices):
    combinations = list(product([0, 1], repeat=3))

    min_ang_sum = float('inf')
    min_combination = []
    for combination in combinations:
        v1 = dual_craters[closest_points_indices[0]][combination[0]].norm
        v2 = dual_craters[closest_points_indices[1]][combination[1]].norm
        v3 = dual_craters[closest_points_indices[2]][combination[2]].norm

        # TODO: does this have to be cubic spline energy?
        ang_sum = (angle_between_vectors(v1,v2)+angle_between_vectors(v1,v3)+angle_between_vectors(v3,v2))
        
        if ang_sum < min_ang_sum:
            min_ang_sum = ang_sum
            min_combination = combination
    
    for i, closest_point_ind in enumerate(closest_points_indices):
        true_duals_indices[closest_point_ind] = min_combination[i]

    return true_duals_indices

from scipy.spatial import Delaunay
import copy
from scipy.interpolate import CubicSpline

def cubic_spline_energy(image_points,surface_norms,surface_points):
    delaunay_triangles = Delaunay(image_points)

    total_curvature_energy = 0
    for simplex in delaunay_triangles.simplices:

        triangle_points = np.array(surface_points)[np.array(simplex)]
        triangle_normals = np.array(surface_norms)[np.array(simplex)]

        # Separate x, y, z coordinates of the triangle points
        x, y, z = triangle_points[:, 0], triangle_points[:, 1], triangle_points[:, 2]

        # Use a simple index parameter for spline fitting (e.g., [0, 1, 2] for 3 points)
        parameter = np.array([0, 1, 2])

        # Construct cubic splines in each dimension based on the points' parameterization
        cs_x = CubicSpline(parameter, x)
        cs_y = CubicSpline(parameter, y)
        cs_z = CubicSpline(parameter, z)

        # Calculate the second derivatives of each spline at all three points (parameter values 0, 1, 2)
        d2x = cs_x(parameter, 2)
        d2y = cs_y(parameter, 2)
        d2z = cs_z(parameter, 2)

        # Calculate the curvature vector for each vertex in the triangle
        curvatures = np.vstack((d2x, d2y, d2z)).T

        # Calculate the alignment of the curvature with the corresponding normals
        dot_products = np.einsum('ij,ij->i', curvatures, triangle_normals)
        norm_curvature = np.linalg.norm(curvatures, axis=1)
        norm_normals = np.linalg.norm(triangle_normals, axis=1)

        alignment_energy = np.sum((dot_products / (norm_curvature * norm_normals))**2)

        # Total curvature energy for the triangle, using a weighted combination
        curvature_energy = np.sum(norm_curvature**2) + alignment_energy

        # Accumulate the total curvature energy
        total_curvature_energy += curvature_energy

    total_curvature_energy = total_curvature_energy/len(delaunay_triangles.simplices)

    return total_curvature_energy

def surface_energy_from_conics(craters_dual, image_conics):

    surface_norms = []
    surface_points = []
    image_points = []
    for i,c in enumerate(image_conics):
        x_c, y_c, _, _, _ = conic_matrix_to_ellipse(c)
        image_points.append(np.array([x_c, y_c]))
        surface_norms.append(craters_dual[i].norm)
        surface_points.append(craters_dual[i].get_crater_centre())
    
    total_curvature_energy = cubic_spline_energy(image_points,surface_norms,surface_points)

    return total_curvature_energy

def surface_energy(craters_w, Pm_c):

    surface_norms = []
    surface_points = []
    image_points = []
    for c in craters_w:
        x_c, y_c, _, _, _ = conic_matrix_to_ellipse(conic_from_crater(c, Pm_c))
        image_points.append(np.array([x_c, y_c]))
        surface_norms.append(c.norm)
        surface_points.append(c.get_crater_centre())
    
    total_curvature_energy = cubic_spline_energy(image_points,surface_norms,surface_points)

    return total_curvature_energy

def dual_from_cubic_spline_interpolation(dual_craters, true_duals_indices, new_crater_index, K, craters_world):

    surface_norms = []
    surface_points = []
    true_surface_points = []
    true_surface_normals = []
    image_points = []
    for i, dual_index in enumerate(true_duals_indices):
        if dual_index != None:
            surface_norms.append(dual_craters[i][dual_index].norm)
            surface_points.append(dual_craters[i][dual_index].get_crater_centre())
            projected_crater_centre = dual_craters[i][dual_index].proj_crater_centre(np.hstack((K,np.array([0,0,0]).reshape((3,1)))))
            image_points.append(projected_crater_centre)
            true_surface_points.append(craters_world[i].get_crater_centre())
            true_surface_normals.append(craters_world[i].norm)
        
    true_surface_points.append(craters_world[new_crater_index].norm)
    true_surface_normals.append(craters_world[new_crater_index].get_crater_centre())
    # print(new_crater_index,len(craters_world))
    # print(true_surface_normals)
    # print(surface_norms)
    # print()

    min_curvature_energy = float('inf')
    best_dual_least_energy = 0
    for i in range(len(dual_craters[new_crater_index])):
        dual_crater_option = dual_craters[new_crater_index][i]
        new_image_points = copy.deepcopy(image_points)
        new_surface_norms = copy.deepcopy(surface_norms)
        new_surface_points = copy.deepcopy(surface_points)

        new_image_points.append(dual_crater_option.proj_crater_centre(np.hstack((K,np.array([0,0,0]).reshape((3,1))))))

        new_surface_norms.append(dual_crater_option.norm)
        new_surface_points.append(dual_crater_option.get_crater_centre())
        
        delaunay_triangles = Delaunay(new_image_points)

        total_curvature_energy = 0
        for simplex in delaunay_triangles.simplices:

            triangle_points = np.array(new_surface_points)[np.array(simplex)]
            true_triangle_points = np.array(true_surface_points)[np.array(simplex)]
            true_triangle_normals = np.array(true_surface_normals)[np.array(simplex)]
            triangle_normals = np.array(new_surface_norms)[np.array(simplex)]
            

            # Separate x, y, z coordinates of the triangle points
            x, y, z = triangle_points[:, 0], triangle_points[:, 1], triangle_points[:, 2]
            tx, ty, tz = true_triangle_points[:, 0], true_triangle_points[:, 1], true_triangle_points[:, 2]
            # print("x,y,z:",x,y,z)
            # print()

            # Use a simple index parameter for spline fitting (e.g., [0, 1, 2] for 3 points)
            parameter = np.array([0, 1, 2])

            # Construct cubic splines in each dimension based on the points' parameterization
            cs_x = CubicSpline(parameter, x)
            cs_y = CubicSpline(parameter, y)
            cs_z = CubicSpline(parameter, z)
            tcs_x = CubicSpline(parameter, tx)
            tcs_y = CubicSpline(parameter, ty)
            tcs_z = CubicSpline(parameter, tz)

            # Calculate the second derivatives of each spline at all three points (parameter values 0, 1, 2)
            d2x = cs_x(parameter, 2)
            d2y = cs_y(parameter, 2)
            d2z = cs_z(parameter, 2)
            td2x = tcs_x(parameter, 2)
            td2y = tcs_y(parameter, 2)
            td2z = tcs_z(parameter, 2)

            # Calculate the curvature vector for each vertex in the triangle
            curvatures = np.vstack((d2x, d2y, d2z)).T
            tcurvatures = np.vstack((td2x, td2y, td2z)).T

            # Calculate the alignment of the curvature with the corresponding normals
            dot_products = np.einsum('ij,ij->i', curvatures, triangle_normals)
            norm_curvature = np.linalg.norm(curvatures, axis=1)
            norm_normals = np.linalg.norm(triangle_normals, axis=1)
            tdot_products = np.einsum('ij,ij->i', tcurvatures, true_triangle_normals)
            tnorm_curvature = np.linalg.norm(tcurvatures, axis=1)
            tnorm_normals = np.linalg.norm(true_triangle_normals, axis=1)

            # Calculate the angle-based energy: the closer the curvature is to the normal, the lower the energy
            alignment_energy = np.sum((dot_products / (norm_curvature * norm_normals))**2)
            talignment_energy = np.sum((tdot_products / (tnorm_curvature * tnorm_normals))**2)

            # Total curvature energy for the triangle, using a weighted combination
            curvature_energy = np.sum(norm_curvature**2) + alignment_energy
            tcurvature_energy = np.sum(tnorm_curvature**2) + talignment_energy

            # Accumulate the total curvature energy
            total_curvature_energy += abs(curvature_energy-tcurvature_energy)

        if min_curvature_energy > total_curvature_energy:
            min_curvature_energy = total_curvature_energy
            best_dual_least_energy = i

    true_duals_indices[new_crater_index] = best_dual_least_energy
    return true_duals_indices

def smooth_dual_from_cubic_spline_interpolation(dual_craters, true_duals_indices, new_crater_index, K):

    surface_norms = []
    surface_points = []
    image_points = []
    for i, dual_index in enumerate(true_duals_indices):
        if dual_index != None:
            surface_norms.append(dual_craters[i][dual_index].norm)
            surface_points.append(dual_craters[i][dual_index].get_crater_centre())
            projected_crater_centre = dual_craters[i][dual_index].proj_crater_centre(np.hstack((K,np.array([0,0,0]).reshape((3,1)))))
            image_points.append(projected_crater_centre)

    min_curvature_energy = float('inf')
    best_dual_least_energy = 0
    for i in range(len(dual_craters[new_crater_index])):
        dual_crater_option = dual_craters[new_crater_index][i]
        new_image_points = copy.deepcopy(image_points)
        new_surface_norms = copy.deepcopy(surface_norms)
        new_surface_points = copy.deepcopy(surface_points)

        new_image_points.append(dual_crater_option.proj_crater_centre(np.hstack((K,np.array([0,0,0]).reshape((3,1))))))

        new_surface_norms.append(dual_crater_option.norm)
        new_surface_points.append(dual_crater_option.get_crater_centre())
        
        delaunay_triangles = Delaunay(new_image_points)

        total_curvature_energy = 0
        for simplex in delaunay_triangles.simplices:

            triangle_points = np.array(new_surface_points)[np.array(simplex)]
            triangle_normals = np.array(new_surface_norms)[np.array(simplex)]

            # Separate x, y, z coordinates of the triangle points
            x, y, z = triangle_points[:, 0], triangle_points[:, 1], triangle_points[:, 2]
            # print("x,y,z:",x,y,z)
            # print()

            # Use a simple index parameter for spline fitting (e.g., [0, 1, 2] for 3 points)
            parameter = np.array([0, 1, 2])

            cs = CubicSpline(parameter, triangle_points)

            # Calculate the curvature vector for each vertex in the triangle
            curvatures = cs(parameter,2)

            # Calculate the alignment of the curvature with the corresponding normals
            dot_products = np.einsum('ij,ij->i', curvatures, triangle_normals)
            norm_curvature = np.linalg.norm(curvatures, axis=1)
            norm_normals = np.linalg.norm(triangle_normals, axis=1)

            # Calculate the angle-based energy: the closer the curvature is to the normal, the lower the energy
            alignment_energy = np.sum((dot_products / (norm_curvature * norm_normals))**2)

            # Total curvature energy for the triangle, using a weighted combination
            curvature_energy = np.sum(norm_curvature**2) + alignment_energy

            # Accumulate the total curvature energy
            total_curvature_energy += curvature_energy

        if min_curvature_energy > total_curvature_energy:
            min_curvature_energy = total_curvature_energy
            best_dual_least_energy = i

    true_duals_indices[new_crater_index] = best_dual_least_energy
    return true_duals_indices






# Maass' method for finding the crater centre from the two dual craters.
# This method firstly chooses the three craters in close proximity and selects the maximally aligned crater norms.
# It then selects the dual of the next crater that minimises the curvature energy of interpolating a cubic spline surface over the 
# crater points already selected (using the Delaunay triangulation over the image points).

def find_crater_norms_from_cubic_spline_interpolation(craters_world, dual_craters, K, is_pangu, scale_data=False):
    if scale_data:
        # Scale all points.
        # Scale everything by the largest distance from a crater to the centre of the moon.
        crater_distances = []
        for crater in craters_world:
            crater_distances.append(np.linalg.norm([crater.X, crater.Y, crater.Z]))
        scale = max(crater_distances)
        offset = 0
        scaled_craters_world = []
        scaled_dual_craters = []
        for i, crater in enumerate(craters_world):
            if not is_pangu:
                scaled_craters_world.append(Crater_w_scaled((crater.X - offset)/scale, (crater.Y - offset)/scale, (crater.Z - offset)/scale, (crater.a)/scale, (crater.b)/scale, crater.phi,crater.id,is_pangu, crater.norm))
            else:
                scaled_craters_world.append(Crater_w_scaled((crater.X - offset)/scale, (crater.Y - offset)/scale, (crater.Z - offset)/scale, (crater.a)/scale, (crater.b)/scale, crater.phi,crater.id,is_pangu=is_pangu))

            crater_dual = dual_craters[i]
            scaled_dual_craters.append((Crater_dual_c((crater_dual[0].X - offset)/scale, (crater_dual[0].Y - offset)/scale, (crater_dual[0].Z - offset)/scale, (crater_dual[0].r)/scale, crater_dual[0].norm), Crater_dual_c((crater_dual[1].X - offset)/scale, (crater_dual[1].Y - offset)/scale, (crater_dual[1].Z - offset)/scale, (crater_dual[1].r)/scale, crater_dual[1].norm)))    


        # Used to mark which crater dual has been selected.
        true_duals_indices = [None]*len(scaled_dual_craters)

        # Might be wrong implementation but I don't think it will affect it much.
        closest_points_indices = closest_three_points_indices(scaled_craters_world)

        true_duals_indices = max_alignment(scaled_dual_craters, closest_points_indices, true_duals_indices)

        for i, dual_index in enumerate(true_duals_indices):
            if dual_index == None:
                true_duals_indices = smooth_dual_from_cubic_spline_interpolation(scaled_dual_craters, true_duals_indices, i, K)
                # true_duals_indices = dual_from_cubic_spline_interpolation(scaled_dual_craters, true_duals_indices, i, K, scaled_craters_world)
                
    
    else:
         # Used to mark which crater dual has been selected.
        true_duals_indices = [None]*len(dual_craters)

        # Might be wrong implementation but I don't think it will affect it much.
        closest_points_indices = closest_three_points_indices(craters_world)

        true_duals_indices = max_alignment(dual_craters, closest_points_indices, true_duals_indices)

        for i, dual_index in enumerate(true_duals_indices):
            if dual_index == None:
                true_duals_indices = smooth_dual_from_cubic_spline_interpolation(dual_craters, true_duals_indices, i, K)
                # true_duals_indices = dual_from_cubic_spline_interpolation(dual_craters, true_duals_indices, i, K, craters_world)

    true_duals = []
    for i, dual_crater in enumerate(dual_craters):
        true_duals.append(dual_crater[true_duals_indices[i]])
        # print(dual_crater[true_duals_indices[i]].get_crater_centre(),dual_crater[true_duals_indices[i]].norm)

    return true_duals, true_duals_indices

   
    

# For more detailed implementation of obtaining the dual circle, look at MATLAB implementation under "crater_centroid_projection/"
# or read paper Homography from Conic Intersection: Camera Calibration based on Arbitrary Circular Patterns
# TODO:(Sofia) remove Pm_c
# TODO:(Sofia) I am using the reprojection of the craters in the WRF with the true pose instead of the image conics because the data right now deals with craters that aren't circles. I need to change this is Crater.py too.
def dual_crater_from_imaged_ellipse(image_conics, craters_world, K, Pm_c, continuous = False):
    dual_craters = []
    for i, crater_w in enumerate(craters_world):
        cam_f = K[0,0]
        cam_wh = K[0,2]

        extrinsics = np.dot(np.linalg.inv(K), Pm_c)
        Tm_c = extrinsics[0:3,0:3]
        Tm_c_euler = R.from_matrix(Tm_c).as_euler('zyx', degrees=True)
        rc = extrinsics[0:3, 3].reshape((3,1))
        rm = -1*np.dot(np.linalg.inv(Tm_c), rc)

        if continuous:
            C = conic_from_crater(crater_w, Pm_c)
        else:
            C = image_conics[i] # conic_from_crater(crater_w, Pm_c) #TODO:(Sofia) this has to be changed to image_conics, but make sure it is a projected circle (not projected ellipse assuming elliptical crater)
        x_c, y_c, a, b, phi = conic_matrix_to_ellipse(C)


        h = x_c-cam_wh
        k = y_c-cam_wh
        A = np.cos(phi)**2/a**2 + np.sin(phi)**2/b**2
        B = (2*np.cos(phi)*np.sin(phi))/a**2 - (2*np.cos(phi)*np.sin(phi))/b**2
        C = np.cos(phi)**2/b**2 + np.sin(phi)**2/a**2
        D = (2*k*np.cos(phi)*np.sin(phi))/b**2 - (2*h*np.sin(phi)**2)/b**2 - (2*k*np.cos(phi)*np.sin(phi))/a**2 - (2*h*np.cos(phi)**2)/a**2
        E = (2*h*np.cos(phi)*np.sin(phi))/b**2 - (2*k*np.sin(phi)**2)/a**2 - (2*h*np.cos(phi)*np.sin(phi))/a**2 - (2*k*np.cos(phi)**2)/b**2
        F = (h**2*np.cos(phi)**2)/a**2 + (k**2*np.cos(phi)**2)/b**2 + (h**2*np.sin(phi)**2)/b**2 + (k**2*np.sin(phi)**2)/a**2 + (2*h*k*np.cos(phi)*np.sin(phi))/a**2 - (2*h*k*np.cos(phi)*np.sin(phi))/b**2 - 1
        Q = np.array([[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, F]])
        
        view_direction = np.array([0,0,1]); # Always along positive z axis.

        kk = np.diag([1, 1, 1/cam_f])
        # K = np.eye(3)

        # Elliptical cone of the projected ellipse through the image plane.
        # cone = transpose([x;y;z])*transpose(K)*Q*K*[x;y;z];

        [val, vec] = eig(np.dot(np.transpose(kk),np.dot(Q,kk)))
        # [val, vec] = eig(np.dot(np.transpose(K),np.dot(C,K)))

        vec = np.transpose(vec)

        for i in range(3):
            vec[i,:] = vec[i,:]/np.linalg.norm(vec[i,:])
        
        sort_order = np.argsort(val)[::-1] # Descending.
        val = val[sort_order]
        #  If lambda 1 and lambda 2 are < 0, then we change the sorting order
        if val[1] < 0:
            sort_order = np.argsort(val)
            val = val[sort_order]
        
        V = vec[sort_order,:]
        
        M = math.sqrt(-val[2]/val[0])
        N = math.sqrt(-val[2]/val[1])

        dual_back_proj_craters = []
        for ii in [-1,1]:
            for jj in [-1,1]:
                aa = ii*math.sqrt((-1*(M-N)*(M+N))/(M**2+1))/N
                bb = 0
                cc = jj*math.sqrt(1-aa)*math.sqrt(aa+1)
                dd = ((1/math.sqrt(2))*crater_w.a*(-((cc + M*aa)*(cc - M*aa)*(cc + N*bb)*(cc - N*bb)*(M**2*N**2*aa**2 - N**2*cc**2 - M**2*cc**2 + M**2*N**2*bb**2 + M**2*N**2*cc**2*((M**4*N**4*aa**4 - 2*M**4*N**4*aa**2*bb**2 + M**4*N**4*bb**4 + 2*M**4*N**2*aa**2*cc**2 - 2*M**4*N**2*bb**2*cc**2 + M**4*cc**4 - 2*M**2*N**4*aa**2*cc**2 + 2*M**2*N**4*bb**2*cc**2 - 2*M**2*N**2*cc**4 + N**4*cc**4)/(M**4*N**4*cc**4))**(1/2)))/((cc**2 + M*N*aa*bb)*(cc**2 - M*N*aa*bb)))**(1/2))/(M*N)

                # Circle norm.
                n = np.array([aa,bb,cc])
                
                #  Circle centre.
                h_c = -(2*aa*dd*(2/N**2 - (2*bb**2)/cc**2))/(cc**2*(4/M**2 - (4*aa**2)/cc**2)*(1/N**2 - bb**2/cc**2))
                k_c = -(2*bb*dd*(2/M**2 - (2*aa**2)/cc**2))/(cc**2*(4/M**2 - (4*aa**2)/cc**2)*(1/N**2 - bb**2/cc**2))
                j_c = -(aa*h_c-dd+bb*k_c)/cc

            
                # Rotated circle centre point
                centre_point = np.dot(np.transpose(V),np.array([h_c, k_c, j_c]))
                #  Rotated circle norm vector
                crater_normal = np.dot(np.transpose(V),n)

                # Check that norm of the vector is in the camera's view, else
                # discard it.
                crater_to_camera_v = np.array([0-centre_point[0],0-centre_point[1],0-centre_point[2]])
                crater_to_camera_v = crater_to_camera_v/np.linalg.norm(crater_to_camera_v)
                angle_crater_view = angle_3D_vectors(view_direction,-crater_to_camera_v);
                if angle_crater_view < math.pi/2:
                    # Set the orientation of normal vectors.
                    angle = angle_3D_vectors(crater_normal,crater_to_camera_v);
                    if angle <= math.pi/2 and angle >= 0:
                        ci1 = Crater_dual_c(centre_point[0], centre_point[1], centre_point[2], crater_w.a, crater_normal)
                        dual_back_proj_craters.append(ci1)
                    else:
                        ci2 = Crater_dual_c(centre_point[0], centre_point[1], centre_point[2], crater_w.a, -1*crater_normal)
                        dual_back_proj_craters.append(ci2)
        dual_craters.append(dual_back_proj_craters) 

        cp1 = (np.dot(np.transpose(Tm_c),np.array([dual_back_proj_craters[0].X,dual_back_proj_craters[0].Y,dual_back_proj_craters[0].Z]).reshape((3,1))) + rm)
        R_norm1 = np.dot(np.transpose(Tm_c),dual_back_proj_craters[0].norm)
        cp2 = (np.dot(np.transpose(Tm_c),np.array([dual_back_proj_craters[1].X,dual_back_proj_craters[1].Y,dual_back_proj_craters[1].Z]).reshape((3,1))) + rm)
        R_norm2 = np.dot(np.transpose(Tm_c),dual_back_proj_craters[1].norm)
        # Validates that a crater in the WRF has the same location as one of the dual craters.
        # print("Diff:",np.linalg.norm(np.array([crater_w.X,crater_w.Y,crater_w.Z]).reshape((3,1))-cp1),np.linalg.norm(np.array([crater_w.X,crater_w.Y,crater_w.Z]).reshape((3,1))-cp2))
        
    return dual_craters
    
def angle_3D_vectors(v1, v2):
    angle_rad = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    return angle_rad

# Dual crater in camera reference frame.
class Crater_dual_c:
    def __init__(self, X, Y, Z, r, norm):
        # 3D location in camera reference frame.
        self.X = X
        self.Y = Y
        self.Z = Z
        self.phi = 0

        # Crater radius (should be same as matched world crater radius)
        self.r = r

        # Conic.
        A = r**2*((math.sin(self.phi))**2)+r**2*((math.cos(self.phi))**2)
        B = 2*(r**2-r**2)*math.cos(self.phi)*math.sin(self.phi)
        C = r**2*((math.cos(self.phi))**2)+r**2*((math.sin(self.phi))**2)
        D = -2*A*0-B*(0)
        E = -B*0-2*C*(0)
        F = A*0**2+B*0*(0)+C*(0)**2-r**2*r**2
        self.conic_matrix_local = np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]])

        # Crater norm
        self.norm = norm

    def get_details(self):
        return [self.X,self.Y,self.Z,self.r,self.phi]

    # Crater centre is on the plane of the crater rim.
    def get_crater_centre(self):
        return np.array([self.X, self.Y, self.Z])
    def get_crater_centre_hom(self):
        return np.array([self.X, self.Y, self.Z, 1])
    
    def proj_crater_centre(self, K_extrinsic_matrix, add_noise=False, mu=0, sigma=0):
        proj_centre = np.dot(K_extrinsic_matrix,self.get_crater_centre_hom())
        if (add_noise):
            return(np.array([proj_centre[0]/proj_centre[2]+random.uniform(-sigma,sigma), proj_centre[1]/proj_centre[2]+random.uniform(-sigma,sigma)]))
        else:
            return(np.array([proj_centre[0]/proj_centre[2], proj_centre[1]/proj_centre[2]]))

class Crater_w_scaled:
    # NOTE: this has been changed to deal with circles ONLY. Uncomment if you want the ellipse crater representation.
    def __init__(self, X, Y, Z, a, b, phi,id="0-0",is_pangu=True, norm=np.array([0,0,1])):
        self.X = (X)
        self.Y = (Y)
        self.Z = (Z)
        self.a = a
        self.b = a
        # self.b = b
        self.phi = 0
        # self.phi = phi
        self.id = id.rstrip().lstrip()
        self.is_pangu = is_pangu

        A = self.a**2*((math.sin(self.phi))**2)+self.b**2*((math.cos(self.phi))**2)
        B = 2*(self.b**2-self.a**2)*math.cos(self.phi)*math.sin(self.phi)
        C = self.a**2*((math.cos(self.phi))**2)+self.b**2*((math.sin(self.phi))**2)
        D = -2*A*0-B*(0)
        E = -B*0-2*C*(0)
        F = A*0**2+B*0*(0)+C*(0)**2-self.a**2*self.b**2

        self.conic_matrix_local = np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]])

        if not self.is_pangu:
            self.norm = norm
        else:
            self.norm = self.get_crater_centre()/np.linalg.norm(self.get_crater_centre())

    # Get a local east, north, up reference frame for each crater.
    def get_ENU(self):
        if np.array_equal(self.norm, np.array([0,0,1])):
            return np.eye(3)
        k = np.array([0, 0, 1])
        u = self.norm
        e = np.cross(k, u)/np.linalg.norm(np.cross(k, u))
        n = np.cross(u, e)/np.linalg.norm(np.cross(u, e))
        TE_M = np.transpose(np.array([e, n, u]))
        return TE_M

    # Crater centre is on the plane of the crater rim.
    def get_crater_centre(self):
        return np.array([self.X, self.Y, self.Z])
    def get_crater_centre_hom(self):
        return np.array([self.X, self.Y, self.Z, 1])
    
    def proj_crater_centre(self, K_extrinsic_matrix, add_noise=False, mu=0, sigma=0):
        proj_centre = np.dot(K_extrinsic_matrix,self.get_crater_centre_hom())
        if (add_noise):
            return(np.array([proj_centre[0]/proj_centre[2]+random.uniform(-sigma,sigma), proj_centre[1]/proj_centre[2]+random.uniform(-sigma,sigma)]))
        else:
            return(np.array([proj_centre[0]/proj_centre[2], proj_centre[1]/proj_centre[2]]))
    def get_details(self):
        return [self.X, self.Y, self.Z, self.a, self.phi]
    def details(self):
        print("X, Y, Z, a, b, phi: ", self.X, self.Y, self.Z, self.a, self.b, self.phi)