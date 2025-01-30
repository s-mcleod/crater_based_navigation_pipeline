import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation as R

def get_extrinsic_matrix_from_pangu(pose, K):
    R_w_ci_intrinsic = R.from_euler('ZXZ',np.array([0,-90,0]),degrees=True).as_matrix()
    R_ci_cf_intrinsic = R.from_euler('ZXZ',np.array([pose.yaw, pose.pitch, 0]),degrees=True).as_matrix()
    R_c_intrinsic = np.dot(R_ci_cf_intrinsic, R_w_ci_intrinsic)
    R_w_c_extrinsic = np.linalg.inv(R_c_intrinsic)
    R_c_roll_extrinsic = R.from_euler('xyz',np.array([0,0,pose.roll]),degrees=True).as_matrix()
    R_w_c = np.dot(R_c_roll_extrinsic,R_w_c_extrinsic)

    Tm_c = R_w_c

    position = ([pose.x, pose.y, pose.z])
    rm = np.array(list(position)) # position of camera in the moon reference frame
    rc = np.dot(Tm_c, -1*rm) # position of camera in the camera reference frame
    so3 = np.empty([3,4])
    so3[0:3, 0:3] = Tm_c
    so3[0:3,3] = rc 

    R_diff = np.dot(R_w_ci_intrinsic, R.from_euler('ZXZ',np.array([0,pose.pitch,0]),degrees=True).as_matrix().T)
    angle_off_nadir_deg = (180/math.pi) * math.acos(min((np.trace(R_diff)-1),2)/2)
    
    return np.dot(K, so3), angle_off_nadir_deg


def get_extrinsic_matrix_from_pnp(rvec_pnp, tvec_pnp, K):
    # Get the extrinsic callibration.
    # NOTE: Pangu provides the moon reference frame to camera reference frame rotation euler angles and the position of the camera in the moon reference frame
    Tm_c = cv2.Rodrigues(rvec_pnp)[0]

    rc = tvec_pnp.astype('float32') # position of camera in the camera reference frame
    so3 = np.empty([3,4])
    so3[0:3, 0:3] = Tm_c
    so3[0:3,3] = rc 
    return np.dot(K, so3)

def get_projection_matrix(R, position_world, K):
    rc = -1*np.dot(R, position_world).astype('float32').reshape(3)
    so3 = np.empty([3,4])
    so3[0:3, 0:3] = R
    so3[0:3,3] = rc 
    return np.dot(K, so3)