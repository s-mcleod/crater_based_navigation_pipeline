import math
import numpy as np
import random 

from src.conics import *

# Crater defined in the world reference frame.
class Crater_w:
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
    
# Crater detected in the camera reference frame. (image reference frame)
class Crater_c:
    def __init__(self, x, y, a, b, phi,id="0-0"):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.phi = phi
        self.id = id.rstrip().lstrip()

        self.conic_matrix_local = ellipse_to_conic_matrix(self.x, self.y, self.a, self.b, self.phi)

    def get_crater_centre(self):
        return [self.x, self.y]
    