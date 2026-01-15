import numpy as np
from src.Motion import wrap_angle


def meas_calc(landmark_pos, state):
    x_pos, y_pos, theta = state
    land_x, land_y = landmark_pos

    r_meas = np.sqrt((land_x - x_pos)**2 + (land_y - y_pos)**2)
    bearing = wrap_angle(np.arctan2(land_y - y_pos, land_x - x_pos) - theta)

    return r_meas, bearing

    
    




