import copy
import numpy as np

from src.Crater import *

# Scale the data by the moon's radius.
def set_scaled_selenographic_data(pose, craters_world, position_bound, propagated_position, is_pangu=True):
    # Scale everything by the largest distance from a crater to the centre of the moon.
    crater_distances = []
    for crater in craters_world:
        crater_distances.append(np.linalg.norm([crater.X, crater.Y, crater.Z]))
    scale = max(crater_distances)
    offset = 0

    # Scale the data by the moon's radius.
    scaled_pose = copy.deepcopy(pose)
    scaled_pose.x = (pose.x - offset)/scale
    scaled_pose.y = (pose.y - offset)/scale
    scaled_pose.z = (pose.z - offset)/scale

    # Scale the propagated pose.
    if (propagated_position):
        scaled_propagated_position = copy.deepcopy(propagated_position)
        scaled_propagated_position = [(p-offset)/scale for p in scaled_propagated_position]
    else:
        scaled_propagated_position = None

    scaled_craters = []
    for crater in craters_world:
        if not is_pangu:
            scaled_craters.append(Crater_w((crater.X - offset)/scale, (crater.Y - offset)/scale, (crater.Z - offset)/scale, (crater.a)/scale, (crater.b)/scale, crater.phi,crater.id,is_pangu, crater.norm))
        else:
            scaled_craters.append(Crater_w((crater.X - offset)/scale, (crater.Y - offset)/scale, (crater.Z - offset)/scale, (crater.a)/scale, (crater.b)/scale, crater.phi,crater.id,is_pangu=is_pangu))

    scaled_position_bound = (position_bound)/scale

    return scaled_pose, scaled_craters, scaled_position_bound, scaled_propagated_position, offset, scale
    