from src.noise_models import build_process_noise, build_meas_noise
import numpy as np
import pandas as pd
from src.step import load_command_data
from src.step import load_ground_truth
from src.step import load_landmark_pos
from src.step import load_measurements_data
from src.Motion import arc_step
from src.Motion import wrap_angle
from src.Measurement import meas_calc
from src.step import simulate_sequence
from src.step import time_align_splice
import scipy.interpolate as interp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

''' This is all a little bit cheaty. HA. But it is calculating the Q and R using the actual
ground truth path. So the calculated covariances and noise terms from these should produce
the optimal model.
'''
def load_barcodes(path):
    barcodes = pd.read_csv(path, delim_whitespace = True, skiprows = 4, header = None, names = ['subject', 'barcode'])
    return barcodes

def time_align_splice_meas(ground, meas): #figure out overlap and cut down dataframes
    from src.Motion import wrap_angle
    meas_span = [meas['time'].iloc[0], meas['time'].iloc[-1]]

    ground_splicer = (ground['time'] >= meas_span[0]) & (ground['time'] < meas_span[1])
    ground['time'] = ground['time'] - meas['time'][0]
   
    ground_spliced = ground[ground_splicer]
    ground_ordered = ground_spliced[['time', 'x_pos', 'y_pos', 'angle']]

    meas['time'] = meas['time'] - meas['time'][0]
    
    return ground_ordered, meas

def get_init_Q():
    ground_truth, commands = time_align_splice(load_ground_truth('data/ds1/ds1_Groundtruth.dat'), load_command_data('data/ds1/ds1_Control.dat'))

    state0 = ground_truth[0][0:-1]

    sequence, t = simulate_sequence(state0, commands, arc_step)

    cov_vel, cov_ang = build_process_noise(sequence, ground_truth, t)

    Q_notime = np.diag([cov_vel, cov_vel, cov_ang])

    return Q_notime

def get_init_R():
    ground_truth, measurements = time_align_splice_meas(load_ground_truth('data/ds1/ds1_Groundtruth.dat'), load_measurements_data('data/ds1/ds1_Measurement.dat'))
    landmark_grounds = load_landmark_pos('data/ds1/ds1_Landmark_Groundtruth.dat')
    landmark_bar = load_barcodes('data/ds1/ds1_Barcodes.dat')
    landmark_grounds['barcode'] = landmark_bar['barcode']
    landmark_dict = {row['barcode']: {'pos': (row['x [m]'], row['y [m]'])} for index, row in landmark_grounds.iterrows()}
    
    subjects = landmark_grounds['barcode'].tolist()
    meas_filtered = measurements[measurements['subject'].isin(subjects)]

    R = build_meas_noise(ground_truth, meas_filtered, landmark_dict)

    return R

if __name__ == "__main__":
    get_init_Q()
    get_init_R()