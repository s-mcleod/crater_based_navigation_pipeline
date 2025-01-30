import numpy as np

# Define a pose as a set of euler angles defining the rotation from world 
# to camera reference frame and position in the world reference frame.
class Pose:
    def __init__(self, x, y, z, yaw, pitch, roll):
        self.x = x
        self.y = y
        self.z = z

        self.yaw = yaw
        self.pitch = pitch 
        self.roll = roll
        self.rvec = np.array([self.yaw, self.pitch, self.roll])