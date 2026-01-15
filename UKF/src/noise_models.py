import scipy.interpolate as interp
import numpy as np
from src.Motion import wrap_angle
from src.Measurement import meas_calc
import pandas as pd

'''
This is comparing the ground truth and dead_reckoned paths to optimize Q and R. 
I interpolate across to get the times to align properly.
'''

def rotate_to_body(theta, world_pos):
    R = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    body_pos = R @ world_pos
    return body_pos

def build_process_noise(dead_state, ground_truth, t_dead): #Compare ground truth and dead reckoned path to extrapolate noise terms
    x_true, y_true, theta_true, ground_time = zip(*ground_truth)
    x_pos, y_pos, theta = zip(*dead_state)

    x_interp = interp.interp1d(ground_time, x_true)
    y_interp = interp.interp1d(ground_time, y_true)
    theta_interp = interp.interp1d(ground_time, theta_true)

    x_matched = x_interp(t_dead[1:-1])
    y_matched = y_interp(t_dead[1:-1])
    theta_matched = theta_interp(t_dead[1:-1])

    deadx_diff = np.diff(x_pos[1:-1])
    deady_diff = np.diff(y_pos[1:-1])
    deadtheta_diff = np.diff(theta[1:-1])
    groundx_diff = np.diff(x_matched)
    groundy_diff = np.diff(y_matched)
    groundtheta_diff = np.diff(theta_matched)

    pos_res = np.array([deadx_diff - groundx_diff, deady_diff - groundy_diff])
    theta_res = wrap_angle(np.array([deadtheta_diff - groundtheta_diff]))

    res_rot = np.empty(pos_res.shape)

    for i in range(len(theta_matched[0:-1])):
        res_rot[:, i] = rotate_to_body(theta_matched[i], pos_res[:, i])

    forward_res = res_rot[0, :]

    dt = round(np.diff(t_dead)[0], 3)

    cov_vel = np.var(forward_res)/dt
    cov_ang = np.var(theta_res)/dt

    return cov_vel, cov_ang

def build_meas_noise(ground_truth, measurements, landmark_dict):

    x_true = np.array(ground_truth['x_pos'])
    y_true = np.array(ground_truth['y_pos'])
    theta_true = np.array(ground_truth['angle'])
    ground_time = np.array(ground_truth['time'])

    x_interpf = interp.interp1d(ground_time, x_true)
    y_interpf = interp.interp1d(ground_time, y_true)
    theta_interpf = interp.interp1d(ground_time, theta_true)

    meas_pred = []
    meas_act = []
    for index, row in measurements.iloc[2:-1].iterrows():
        landmark_pos = landmark_dict[row['subject']]['pos']
        pose = (x_interpf(row['time']), y_interpf(row['time']), theta_interpf(row['time']))
        meas_pred.append(meas_calc(landmark_pos, pose))
        meas_act.append((row['range'], row['bearing']))
    
    meas_pred_df = pd.DataFrame(meas_pred, columns = ['range', 'bearing'])
    meas_act_df = pd.DataFrame(meas_act, columns = ['range', 'bearing'])

    range_res = np.array(meas_act_df['range'] - meas_pred_df['range'])
    bearing_res = np.array(wrap_angle(meas_act_df['bearing'] - meas_pred_df['bearing']))


    res_array = np.stack((range_res, bearing_res), axis = 1)

    covariance = np.cov(res_array, rowvar = False)

    return covariance



