from src.step import load_ground_truth
from src.step import load_command_data
from src.step import load_measurements_data
from src.step import load_landmark_pos
from src.tuning import load_barcodes
from src.plots import plot_path_UKF
from src.Motion import wrap_angle
from src.Measurement import meas_calc
from src.plots import plot_path
import numpy as np
import pandas as pd
from src.errorcalc import diff_calc_UKF
from src.plots import plot_pos_diff_UKF
from src.plots import plot_innovation
from src.tuning import get_init_Q
from src.tuning import get_init_R


def draw_sigma(P, state, lambda_): #Drawing my sigma points
    n = P.shape[0]
    sigma_points = np.empty((2*n + 1, n))
    sigma_points[0] = state

    cholesk = (np.sqrt(n + lambda_) * np.linalg.cholesky(P))
    
    for i in range(1, n+1):
        sigma_points[i] = state + cholesk[:, i-1]
    
    for i in range(n + 1, (2*n)+1):
        sigma_points[i]  =  state - cholesk[:, i-n-1]

    return sigma_points



def initialize(Q_scale, R_scale): # Just initializing my values/weights and getting the data
    ground_truth = load_ground_truth('data/ds1/ds1_Groundtruth.dat')
    commands = load_command_data('data/ds1/ds1_Control.dat')
    measurements = load_measurements_data('data/ds1/ds1_Measurement.dat')
   
    state0 = ground_truth.iloc[0][1::]
   
    alpha = 1e-3
    beta = 2
    kappa = 0
    n = len(state0)
    
    lambda_ = alpha**2 * (n + kappa) - n

    Q = np.array([0.001, .001, .0001]) * Q_scale
    R = np.array([.5, .1]) * R_scale
    P = np.array([.01, .01, .001])
    Q_mat = np.eye(3) * Q
    R_mat = np.eye(2) * R
    P_mat = np.eye(3) * P
    
    W_m = np.empty((2*n +1))
    W_c = np.empty((2*n + 1))

    W_m[0] = lambda_/(n + lambda_)
    W_c[0] = lambda_/(n + lambda_) + (1 - alpha**2 + beta)

    for i in range(1, (2*n)+1):
        W_m[i] = 1/(2 * (n + lambda_))
        W_c[i] = 1/(2 * (n + lambda_))

    meas_rebase, command_rebase, ground_rebase = time_splice_UKF(measurements, commands, ground_truth) #this just aligns all the data

    print(W_m.shape)
    return meas_rebase, command_rebase, ground_rebase, Q_mat, R_mat, P_mat, W_m, W_c, n, lambda_, state0

def tuned_initialize(): #The tuned model
    ground_truth = load_ground_truth('data/ds1/ds1_Groundtruth.dat')
    commands = load_command_data('data/ds1/ds1_Control.dat')
    measurements = load_measurements_data('data/ds1/ds1_Measurement.dat')
   
    state0 = ground_truth.iloc[0][1::]
   
    alpha = 1e-3
    beta = 2
    kappa = 0
    n = len(state0)
    
    lambda_ = alpha**2 * (n + kappa) - n

    P = np.array([.01, .01, .001])
    Q_mat = get_init_Q()
    R_mat = get_init_R()
    P_mat = np.eye(3) * P
    
    W_m = np.empty((2*n +1))
    W_c = np.empty((2*n + 1))

    W_m[0] = lambda_/(n + lambda_)
    W_c[0] = lambda_/(n + lambda_) + (1 - alpha**2 + beta)

    for i in range(1, (2*n)+1):
        W_m[i] = 1/(2 * (n + lambda_))
        W_c[i] = 1/(2 * (n + lambda_))

    meas_rebase, command_rebase, ground_rebase = time_splice_UKF(measurements, commands, ground_truth)

    print(W_m.shape)
    return meas_rebase, command_rebase, ground_rebase, Q_mat, R_mat, P_mat, W_m, W_c, n, lambda_, state0

def arc_step_UKF(state, control, dt): #my update for each time step: motion model
    x, y, theta = state
    v, omega = control

    if abs(omega) < 1e-6:
        x_new = x + v * dt * np.cos(theta)
        y_new = y + v * dt * np.sin(theta)
        theta_new = theta
    
    else:
        x_new = x + (v/omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        y_new = y - (v/omega) * (np.cos(theta + omega * dt) - np.cos(theta))
        theta_new = wrap_angle(theta + omega * dt)
    
    return np.array([x_new, y_new, theta_new])

def time_splice_UKF(meas, command, ground): #splicing all to same time intervals, ground included just for graphing
    range_time = [max(min(meas['time']), min(command['time']), min(ground['time'])), min(max(meas['time']), max(command['time']), max(ground['time']))]
    ground_splicer = (ground['time'] >= range_time[0]) & (ground['time'] <= range_time[1])
    command_splicer = (command['time'] >= range_time[0]) & (command['time'] <= range_time[1])
    meas_splicer = (meas['time'] >= range_time[0]) & (meas['time'] <= range_time[1])

    ground_spliced = ground[ground_splicer]
    command_spliced = command[command_splicer]
    meas_spliced = meas[meas_splicer]

    ground_spliced['time'] = ground_spliced['time'] - range_time[0]
    command_spliced['time'] = command_spliced['time'] - range_time[0]
    meas_spliced['time'] = meas_spliced['time'] - range_time[0]

    return meas_spliced, command_spliced, ground_spliced

def time_update(sigmas, W_c, W_m, dt, Q, control): #Stepping through the motion model, assuming no measurment update

    sigma_up = np.empty_like(sigmas)
    for i in range(sigmas.shape[0]):
        sigma_up[i, :] = arc_step_UKF(sigmas[i, :], control, dt)

    sigma_up[:, 2] = wrap_angle(sigma_up[:, 2])
    state_up = np.zeros(sigmas.shape[1])
    thetas = sigma_up[:, 2]
    C = 0
    S = 0

    for i in range(W_m.shape[0]):
        state_up[0:2] += sigma_up[i, 0:2] * W_m[i]
        C += W_m[i] * np.cos(thetas[i])
        S += W_m[i] * np.sin(thetas[i])
    theta_av = np.arctan2(S, C)
    state_up[2] = theta_av #This just ensures a circular mean for theta

    P_up = np.zeros([sigmas.shape[1], sigmas.shape[1]])
    res_mat = sigma_up - state_up
    res_mat[:, 2] = wrap_angle(res_mat[:, 2])
    
    for i in range(W_c.shape[0]):
        P_up += W_c[i] * np.outer(res_mat[i, :], (res_mat[i, :].T))
    
    P_up = P_up + Q * dt

    P_up = (P_up + P_up.T) / 2
    

    return P_up, state_up, sigma_up

def measurement_update(measurement, landmark_dict, W_m, W_c, sigma_points, state, P, R): #applying the measurement model to correct the mean state
    landmark, r_meas, bearing = measurement
    landmark_pos = landmark_dict[int(landmark)]['pos']

    sigma_meas = np.empty([sigma_points.shape[0], 2])
    for i in range(sigma_points.shape[0]):
        sigma_meas[i, :] = meas_calc(landmark_pos, sigma_points[i, :])

    meas_guess = np.zeros(sigma_meas.shape[1])
    thetas = sigma_meas[:, 1]
    C = 0
    S = 0

    for i in range(W_m.shape[0]):
        meas_guess[0] += sigma_meas[i, 0] * W_m[i]
        C += W_m[i] * np.cos(thetas[i])
        S += W_m[i] * np.sin(thetas[i])
    theta_av = np.arctan2(S, C)
    meas_guess[1] = theta_av

    meas_cov = np.zeros([sigma_meas.shape[1], sigma_meas.shape[1]])
    meas_res_mat = sigma_meas - meas_guess
    meas_res_mat[:, 1] = wrap_angle(meas_res_mat[:, 1])
    
    for i in range(W_c.shape[0]):
        meas_cov += W_c[i] * np.outer(meas_res_mat[i, :], (meas_res_mat[i, :].T))

    meas_cov = meas_cov + R

    pos_meas_cov = np.zeros([sigma_points.shape[1], sigma_meas.shape[1]])
    pos_res_mat = sigma_points - state
    pos_res_mat[:, 2] = wrap_angle(pos_res_mat[:, 2])

    for i in range(W_c.shape[0]):
        pos_meas_cov += W_c[i] * np.outer(pos_res_mat[i, :], meas_res_mat[i, :].T)

    meas = np.array([r_meas, bearing])
    K_gain = pos_meas_cov @ np.linalg.inv(meas_cov)
    innov = meas - meas_guess
    innov[1] = wrap_angle(innov[1])

    state_corr = state + np.matmul(K_gain, innov)
    state_corr[2] = wrap_angle(state_corr[2])

    P_corr = P - K_gain @ meas_cov @ K_gain.T

    P_corr = (P_corr + P_corr.T) / 2 #small correction to P just to symmetrize

    return state_corr, P_corr, innov


def UKF_tuned_simulate(): #Just the tuned variation of the simulation
    meas_rebase, command_rebase, ground_rebase, Q_mat, R_mat, P_mat, W_m, W_c, n, lambda_, state0 = tuned_initialize()

    landmark_grounds = load_landmark_pos('data/ds1/ds1_Landmark_Groundtruth.dat')
    landmark_bar = load_barcodes('data/ds1/ds1_Barcodes.dat')
    landmark_grounds['barcode'] = landmark_bar['barcode']
    landmark_dict = {row['barcode']: {'pos': (row['x [m]'], row['y [m]'])} for index, row in landmark_grounds.iterrows()}
    
    subjects = landmark_grounds['barcode'].tolist()
    meas_filtered = meas_rebase[meas_rebase['subject'].isin(subjects)]

    measures = []
    commands = []
    for index, row in meas_filtered.iterrows():
        measures.append((row['subject'], row['range'], row['bearing']))
    for index, row in command_rebase.iterrows():
        commands.append((row['vel'], row['ang_vel']))

    meas_dict = {'time': meas_filtered['time'], 'info': measures, 'comm_meas': np.zeros_like(meas_filtered['time'])}
    comm_dict = {'time': command_rebase['time'], 'info': commands, 'comm_meas': np.ones_like(command_rebase['time'])}

    meas_df = pd.DataFrame(meas_dict)
    comm_df = pd.DataFrame(comm_dict)

    full_df = pd.concat([comm_df, meas_df]).sort_values(by=['time', 'comm_meas'])
    full_df['prev_time'] = full_df['time'].shift(1)
    full_df = full_df.dropna() #Building dataframe that has all necessary info
    
    track_state = [state0]
    state = state0
    control = (0, 0)
    track_innov = []
    
    for index, row in full_df.iterrows(): #This loop just propagates everything through the model
        dt = row['time'] - row['prev_time'] 
        sigma_points = draw_sigma(P_mat, state, lambda_)
        if row['comm_meas'] == 1:
            P_mat, state, sigma_up = time_update(sigma_points, W_c, W_m, dt, Q_mat, control)
            track_state.append(state)
            control = row['info']
        else:
            P_mat, state, sigma_up = time_update(sigma_points, W_c, W_m, dt, Q_mat, control)
            state_corr, P_mat, innov = measurement_update(row['info'], landmark_dict, W_m, W_c, sigma_up, state, P_mat, R_mat)
            state = state_corr
            track_state.append(state)
            track_innov.append([innov, dt, row['time']])


    plot_path_UKF(track_state, ground_rebase[['x_pos', 'y_pos', 'angle', 'time']], full_df['time'], 'UKF vs Ground', 'plots/UKFvGround_trial_tuned')
    diff = diff_calc_UKF(track_state, ground_rebase, full_df['time'])
    plot_pos_diff_UKF(diff, full_df['time'], 'plots/UKFposerror_tuned')
    plot_innovation(track_innov, 'plots/innovationrangevstimes_tuned', 'plots/innovationbearingvstime_tuned')

def UKF_simulate(Q_scale, R_scale):
    meas_rebase, command_rebase, ground_rebase, Q_mat, R_mat, P_mat, W_m, W_c, n, lambda_, state0 = initialize(Q_scale, R_scale)

    landmark_grounds = load_landmark_pos('data/ds1/ds1_Landmark_Groundtruth.dat')
    landmark_bar = load_barcodes('data/ds1/ds1_Barcodes.dat')
    landmark_grounds['barcode'] = landmark_bar['barcode']
    landmark_dict = {row['barcode']: {'pos': (row['x [m]'], row['y [m]'])} for index, row in landmark_grounds.iterrows()}
    
    subjects = landmark_grounds['barcode'].tolist()
    meas_filtered = meas_rebase[meas_rebase['subject'].isin(subjects)]

    measures = []
    commands = []
    for index, row in meas_filtered.iterrows():
        measures.append((row['subject'], row['range'], row['bearing']))
    for index, row in command_rebase.iterrows():
        commands.append((row['vel'], row['ang_vel']))

    meas_dict = {'time': meas_filtered['time'], 'info': measures, 'comm_meas': np.zeros_like(meas_filtered['time'])}
    comm_dict = {'time': command_rebase['time'], 'info': commands, 'comm_meas': np.ones_like(command_rebase['time'])}

    meas_df = pd.DataFrame(meas_dict)
    comm_df = pd.DataFrame(comm_dict)

    full_df = pd.concat([comm_df, meas_df]).sort_values(by=['time', 'comm_meas'])
    full_df['prev_time'] = full_df['time'].shift(1)
    full_df = full_df.dropna()
    
    track_state = [state0]
    state = state0
    control = (0, 0)
    track_innov = []
    
    for index, row in full_df.iterrows():
        dt = row['time'] - row['prev_time'] 
        sigma_points = draw_sigma(P_mat, state, lambda_)
        if row['comm_meas'] == 1:
            P_mat, state, sigma_up = time_update(sigma_points, W_c, W_m, dt, Q_mat, control)
            track_state.append(state)
            control = row['info']
        else:
            P_mat, state, sigma_up = time_update(sigma_points, W_c, W_m, dt, Q_mat, control)
            state_corr, P_mat, innov = measurement_update(row['info'], landmark_dict, W_m, W_c, sigma_up, state, P_mat, R_mat)
            state = state_corr
            track_state.append(state)
            track_innov.append([innov, dt, row['time']])


    plot_path_UKF(track_state, ground_rebase[['x_pos', 'y_pos', 'angle', 'time']], full_df['time'], f'UKF vs Ground', f'plots/UKFvGround_trial{int(Q_scale)},{int(R_scale)}')
    diff = diff_calc_UKF(track_state, ground_rebase, full_df['time'])
    plot_pos_diff_UKF(diff, full_df['time'], f'plots/UKFposerror{int(Q_scale)},{int(R_scale)}')
    plot_innovation(track_innov, f'plots/innovationrangevstimes{int(Q_scale)},{int(R_scale)}', f'plots/innovationbearingvstime{int(Q_scale)},{int(R_scale)}')

def UKF_step2(): #This is just for step 2 to show that it is the same as the one with no model
    meas_rebase, command_rebase, ground_rebase, Q_mat, R_mat, P_mat, W_m, W_c, n, lambda_, state0 = initialize()
    state0 = np.array([0, 0, 0])

    control_dict = {'time': [0, 1, 2, 3, 4, 5, 6], 'info': [(0, 0), (.5, 0), (0, -1/(2*np.pi)), (.5, 0), (0, 1/(2*np.pi)), (.5, 0), (0, 0)], 'comm_meas': [1, 1, 1, 1, 1, 1, 1]}
    full_df = pd.DataFrame(control_dict)
    full_df['prev_time'] = full_df['time'].shift(1)
    full_df = full_df.dropna()
  
    track_state = [state0]
    state = state0

    track_innov = []
    control = (0, 0)
    
    for index, row in full_df.iterrows():
        dt = row['time'] - row['prev_time'] 
        print(dt)
        sigma_points = draw_sigma(P_mat, state, lambda_)
        if row['comm_meas'] == 1:
            P_mat, state, sigma_up = time_update(sigma_points, W_c, W_m, dt, Q_mat, control)
            track_state.append(state)
            control = row['info']
        # else:
        #     P_mat, state, sigma_up = time_update(sigma_points, W_c, W_m, dt, Q_mat, control)
        #     state_corr, P_mat, innov = measurement_update(row['info'], landmark_dict, W_m, W_c, sigma_up, state, P_mat, R_mat)
        #     state = state_corr
        #     track_state.append(state)
        #     track_innov.append([innov, dt, row['time']])
        
    plot_path(track_state, full_df['time'], 'Step 2 Controls with UKF', 'plots/step2UKF')

if __name__ == '__main__':
    initialize()
    UKF_simulate()