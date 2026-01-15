import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
This was all mostly for part A, and I repeated some of functions in UKF to make it easier on myself and because I got better at coding over this project haha
'''

def load_commands():
    coms =[(1, .5, 0, 1), (1, 0, -1/(2*np.pi), 2), (1, .5, 0, 3), (1, 0, 1/(2*np.pi), 4), (1, .5, 0, 5)]
    return coms
 
def load_command_data(path):
    commands = pd.read_csv(path, delim_whitespace=True, skiprows = 4, header = None, names = ['time', 'vel', 'ang_vel'])
    return commands

def load_ground_truth(path): 
    true_positions = pd.read_csv(path, delim_whitespace = True, skiprows = 4, header= None, names = ['time', 'x_pos', 'y_pos', 'angle'])
    return true_positions

def time_align_splice(ground, command): #figure out overlap and cut down dataframes
    from src.Motion import wrap_angle
    command_span = [command['time'].iloc[0], command['time'].iloc[-1]]

    ground_splicer = (ground['time'] >= command_span[0]) & (ground['time'] < command_span[1])
    ground['time'] = ground['time'] - command['time'][0]
   
    ground_spliced = ground[ground_splicer]
    ground_ordered = ground_spliced[['x_pos', 'y_pos', 'angle', 'time']]

    command['time'] = command['time'] - command['time'][0]
    
    duration = command['time'].diff(periods = 1)
    command.insert(1, 'dur', duration)
    com_dur = command[['dur', 'vel', 'ang_vel', 'time']].dropna()

    com_dur_list = com_dur.values.tolist()
    ground_list = ground_ordered.values.tolist()
    
    return ground_list, com_dur_list

def simulate_sequence(init_state, commands, integrator): #Just simulate the robot path
    t = [0]
    states = [init_state]
    for control in commands:
        states.append(integrator(states[-1], control[1:-1], control[0]))
        t.append(control[-1])
    return np.array(states), t

def make_step2_plot(): ##make the plots for step 2
    from src.Motion import arc_step
    from src.plots import plot_path
    
    state0 = (0, 0, 0)
    commands = load_commands()
    sequence, t = simulate_sequence(state0, commands, arc_step)
  
    plot_path(sequence, t, 'Position and Angle of Robot', 'plots/step2plots.png')

def make_step3_plot(): ##make the plots for step 3
    from src.Motion import arc_step
    from src.plots import plot_path_3
    from src.plots import plot_pos_diff
    from src.errorcalc import diff_calc
    
    ground_truth, commands = time_align_splice(load_ground_truth('data/ds1/ds1_Groundtruth.dat'), load_command_data('data/ds1/ds1_Control.dat'))

    state0 = ground_truth[0][0:-1]

    sequence, t = simulate_sequence(state0, commands, arc_step)

    plot_path_3(sequence, ground_truth, t, 'Position and Angle of Robot', 'plots/step3plots.png')
    diff = diff_calc(sequence, ground_truth, t)
    plot_pos_diff(diff, t, 'plots/step3error.png')

def load_landmark_pos(path):
    landmark_pos = pd.read_csv(path, delim_whitespace=True, skiprows = 4, header = None, names = ['subject #', 'x [m]', 'y [m]', 'x std dev', 'y std dev'])
    return landmark_pos

def load_sample_measurements():
    sample_measurement = [(6, 2, 3, 0), (13, 0, 3, 0), (17, 1, -2, 0)]
    return sample_measurement

def load_measurements_data(path):
    measurements = pd.read_csv(path, delim_whitespace=True, skiprows = 4, header = None, names = ['time', 'subject', 'range', 'bearing'])
    return measurements

def step6outputs(): #Calculating the range and bearing using my measurement model.
    from src.Measurement import meas_calc
    measurements = load_sample_measurements()
    landmark_grounds = load_landmark_pos('data/ds1/ds1_Landmark_Groundtruth.dat')
    landmark_dict = {row['subject #']: {'pos': (row['x [m]'], row['y [m]']), 'dev': (row['x std dev'], row['y std dev'])} for index, row in landmark_grounds.iterrows()}

    land_noticed_pos = []
    state = []
    for measure in measurements:
        land_noticed_pos.append(landmark_dict[measure[0]]['pos'])
        state.append(measure[1::]) 

    for j in range(len(land_noticed_pos)):
        r_meas, bearing = meas_calc(land_noticed_pos[j], state[j])
        print(f'\nfor test {j} the calculated range and bearing are {r_meas} and {bearing}')





