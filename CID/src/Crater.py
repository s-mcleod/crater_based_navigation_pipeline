import math
import numpy as np
import random
from numba import float64
from numba.experimental import jitclass
from src.conics import *

# Define the specification for Crater_w
crater_w_spec = [
    ('X', float64),
    ('Y', float64),
    ('Z', float64),
    ('a', float64),
    ('b', float64),
    ('phi', float64),
    ('conic_matrix_local', float64[:, :])
]

# Define the specification for Crater_c
crater_c_spec = [
    ('x', float64),
    ('y', float64),
    ('a', float64),
    ('b', float64),
    ('phi', float64),
    ('conic_matrix_local', float64[:, :])
]



# Crater defined in the world reference frame.
@jitclass(crater_w_spec)
class Crater_w:
    def __init__(self, X, Y, Z, a, b):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.a = a
        self.b = b
        # Assuming circular crater.
        self.phi = 0

        A = a**2*((np.sin(self.phi))**2)+b**2*((np.cos(self.phi))**2)
        B = 2*(b**2-a**2)*np.cos(self.phi)*np.sin(self.phi)
        C = a**2*((np.cos(self.phi))**2)+b**2*((np.sin(self.phi))**2)
        D = -2*A*0-B*(0)
        E = -B*0-2*C*(0)
        F = A*0**2+B*0*(0)+C*(0)**2-a**2*b**2

        self.conic_matrix_local = np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]])

    # Get a local east, north, up reference frame for each crater.
    def get_ENU(self):
        Pc_M = self.get_crater_centre()
        k = np.array([0, 0, 1])
        u = Pc_M/np.linalg.norm(Pc_M)
        e = np.cross(k, u)/np.linalg.norm(np.cross(k, u))
        n = np.cross(u, e)/np.linalg.norm(np.cross(u, e))
        # TE_M = np.transpose(np.array([e, n, u]))
        TE_M = np.empty((3, len(e)))  # Assuming e, n, and u have the same length
        TE_M[0, :] = e
        TE_M[1, :] = n
        TE_M[2, :] = u

        # TE_M = np.array([e, n, u])
        TE_M = TE_M.transpose()
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
    
    
# Crater detected in the camera reference frame.
@jitclass(crater_c_spec)
class Crater_c:
    def __init__(self, x, y, a, b, phi):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.phi = phi

        self.conic_matrix_local = ellipse_to_conic_matrix(self.x, self.y, self.a, self.b, self.phi)

    def get_crater_centre(self):
        return [self.x, self.y]
    
    
    