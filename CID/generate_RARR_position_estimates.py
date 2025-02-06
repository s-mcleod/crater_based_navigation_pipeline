import sys
import numpy as np
import os
import random

def write_position_noise(flight_file, RARR_positions, position_noise):
    f = open(flight_file, 'r')
    lines = f.readlines()
    lines = [i.split() for i in lines]
    camera_extrinsics = np.zeros([len(lines), 3, 4])
    with open(RARR_positions,"w") as file:
        file.write(str(position_noise)+"\n")
        for i, line in enumerate(lines):
            # Camera pose line is prefixed with "start" and has structure -> x, y, z, yaw, pitch, roll respectively
            if len(line) > 0 and line[0] == "start":
                pose = np.float_(line[1:])
                x, y, z, _, _, _ = pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
                x_noise = random.uniform(x-position_noise, x+position_noise)
                y_noise = random.uniform(y-position_noise, y+position_noise)
                z_noise = random.uniform(z-position_noise, z+position_noise)
                
                file.write(str(x_noise)+", "+str(y_noise)+", "+str(z_noise)+"\n")

    return camera_extrinsics

if __name__ == "__main__":
    random.seed(42)
    flight_file = sys.argv[1]
    RARR_positions = sys.argv[2]
    pos_err = float(sys.argv[3])
    write_position_noise(flight_file,RARR_positions,pos_err)