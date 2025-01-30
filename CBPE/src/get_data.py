import cv2
import glob
import math
import numpy as np
import random
import re

from src.Pose import *
from src.Crater import *

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

    return (np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]))

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

# Get a list of 3D craters in the world reference frame.
def get_craters_world(craters_world_file, is_pangu=True):
    f = open(craters_world_file,"r")
    lines = f.readlines()[1:] #ignore the first line
    lines = [(re.split(r',\s*', i)) for i in lines]
    if not is_pangu:
        craters = [Crater_w(float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4]), float(i[5]),i[6],is_pangu,np.array([float(i[7]),float(i[8]),float(i[9])])) for i in lines]
    else:
        craters = [Crater_w(float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4]), float(i[5]),i[6],is_pangu) for i in lines]
    return craters

# Get a list of detected craters in the image plane.
# The camera detected craters can be offset if the add_noise flag is on.
def get_craters_cam(craters_cam_file, add_noise, mu=0, sigma=5):
    f = open(craters_cam_file,"r")
    lines = f.readlines()[1:] #ignore the first line
    lines = [(re.split(r',\s*', i)) for i in lines]
    craters = [Crater_c(float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4]), i[5]) for i in lines]
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

def file_name_deg_off_nadir_dict(dir):
    f = open(dir,"r")
    lines = f.readlines()
    lines = [(line.split(',')) for line in lines]
    dict = {}
    for line in lines:
        dict[line[0][:-4]] = int(line[1])
    return dict