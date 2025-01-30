import math
from mpmath import mp
import numpy as np
import cv2
from src.conics import *

from src.metrics import *
from src.conics import *

def write_conic_to_projected_ellipse_csv_no_return(A, csv_file):
    x_c, y_c, a, b, phi = conic_matrix_to_ellipse(A)
    center_coordinates = (int(x_c), int(y_c))
    axesLength = (int(a), int(b))
    angle = int(phi*180/math.pi)
    ellipse = "("+str(center_coordinates[0])+","+str(center_coordinates[1])+","+str(axesLength[0])+","+str(axesLength[1])+","+str(angle)+")"
    csv_file.write(ellipse)
    return center_coordinates, axesLength, angle

def project_conic(A, image, colour, show_ellipse_centre=False, size = 3, fill=False, intensity = 0):
    x_c, y_c, a, b, phi = conic_matrix_to_ellipse(A)
    # Only store the ellipses which are partially (or fully) in the image frame.
    image_width, image_height, _ = image.shape
    if (a < image_width and a < image_height and b < image_width and b < image_height):
        if (x_c+a >= 0 and x_c-a < image_width and y_c+a >= 0 and y_c-a < image_height):
            # Only project top largest craters for visualisation purposes only.

            center_coordinates = (int(x_c), int(y_c))
            axesLength = (int(a), int(b))
            angle = int(phi*180/math.pi)

            startAngle = 0
            endAngle = 360
            if (fill and intensity == 0):
                image = cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, (0, 0, 0), -1)
            thickness = 2
            image = cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, colour, thickness)
            if show_ellipse_centre:
                image = cv2.circle(image, center_coordinates, size, colour, -1)

            # print(center_coordinates, axesLength, angle)
            
    return image

def project_conics(conics, image, dir, img_filename, colour, show_ellipse_centre, size=3):
    for i in range(len(conics)):
        A = conics[i]
        image = project_conic(A, image, colour, show_ellipse_centre=show_ellipse_centre, size=size)
    cv2.imwrite(dir+img_filename, image)


def project_craters(craters, Pm_c, image, dir, img_filename, colour, fill=False, intensities=[]):
    # image = show_crater_centres(craters, Pm_c, image, colour)
    if not fill:
        intensities = [0]*len(craters)
    # Pm_c is the projection matrix k[R|t].
    # Sort craters from largest to smallest radius.
    # craters.sort(key=lambda x: x.a, reverse=True)
    # craters = craters[:100] #TODO: remove
    for i, c in enumerate(craters):
        A = conic_from_crater(c, Pm_c)
        image = project_conic(A, image, colour, fill, intensities[i])
    cv2.imwrite(dir+img_filename, image)

def project_dual_crater_centres(craters, Pm_c, image, dir, img_filename, colour, fill=False, intensities=[], size = 3):
    image = show_crater_centres(craters, Pm_c, image, colour, size)
    cv2.imwrite(dir+img_filename, image)


def get_projected_crater_centres(craters, Pm_c):
    centre_coordinates = []
    for i, c in enumerate(craters):
        A = conic_from_crater(c, Pm_c)
        x_c, y_c, a, b, phi = conic_matrix_to_ellipse(A)
        centre_coordinates.append([x_c, y_c])
    return centre_coordinates


def show_crater_centres(craters_world, extrinsics, image, colour, size=3):
    for crater in craters_world:
        crater_centre_3D_hom = crater.get_crater_centre_hom()
        proj_crater_centre = np.dot(extrinsics, crater_centre_3D_hom)
        proj_crater_centre_cam = np.array([proj_crater_centre[0]/proj_crater_centre[2], proj_crater_centre[1]/proj_crater_centre[2]])

        # Project the two crater centres.
        image = cv2.circle(image, (int(proj_crater_centre_cam[0]), int(proj_crater_centre_cam[1])), size, colour, -1)
    return image

def proj_centre_loc_dual(crater_dual, Pm_c):
    crater_centre_3D_hom = crater_dual.get_crater_centre_hom()
    proj_crater_centre = np.dot(Pm_c, crater_centre_3D_hom)
    return np.array([proj_crater_centre[0]/proj_crater_centre[2], proj_crater_centre[1]/proj_crater_centre[2]])