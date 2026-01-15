import scipy.interpolate as interp
import numpy as np
from src.Motion import wrap_angle

'''
These are my error calculations for the positions
'''


def diff_calc(dead_state, ground_truth, t_dead):
    x_true, y_true, theta_true, ground_time = zip(*ground_truth)
    x_pos, y_pos, theta = zip(*dead_state)

    x_interp = interp.interp1d(ground_time, x_true)
    y_interp = interp.interp1d(ground_time, y_true)

    x_matched = x_interp(t_dead[1:-1])
    y_matched = y_interp(t_dead[1:-1])

    diff = np.sqrt((x_matched - x_pos[1:-1])**2 + (y_matched - y_pos[1:-1])**2)

    return diff

def diff_calc_UKF(UKF_state, ground_truth, t_controls):
    x_true, y_true, theta_true, ground_time = ground_truth['x_pos'], ground_truth['y_pos'], ground_truth['angle'], ground_truth['time']
    x_pos, y_pos, theta = zip(*UKF_state)


    x_interp = interp.interp1d(ground_time, x_true)
    y_interp = interp.interp1d(ground_time, y_true)


    x_matched = x_interp(t_controls[1:-1])
    y_matched = y_interp(t_controls[1:-1])



    diff = np.sqrt((x_matched - x_pos[1:-2])**2 + (y_matched - y_pos[1:-2])**2)

    return diff