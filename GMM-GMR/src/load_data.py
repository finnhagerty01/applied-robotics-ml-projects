import pandas as pd
import numpy as np
import scipy.interpolate as interp

def load_landmark_pos(path):
    landmark_pos = pd.read_csv(path, delim_whitespace=True, skiprows = 4, header = None, names = ['subject #', 'x [m]', 'y [m]', 'x std dev', 'y std dev'])
    return landmark_pos

def load_measurements_data(path):
    measurements = pd.read_csv(path, delim_whitespace=True, skiprows = 4, header = None, names = ['time', 'subject', 'range', 'bearing'])
    return measurements

def load_ground_truth(path): 
    true_positions = pd.read_csv(path, delim_whitespace = True, skiprows = 4, header= None, names = ['time', 'x_pos', 'y_pos', 'angle'])
    return true_positions

def load_barcodes(path):
    barcodes = pd.read_csv(path, delim_whitespace = True, skiprows = 4, header = None, names = ['subject', 'barcode'])
    return barcodes

def time_splice(meas, ground): #splicing all to same time intervals, ground included just for graphing
    range_time = [max(min(meas['time']), min(ground['time'])), min(max(meas['time']), max(ground['time']))]
    ground_splicer = (ground['time'] >= range_time[0]) & (ground['time'] <= range_time[1])
    meas_splicer = (meas['time'] >= range_time[0]) & (meas['time'] <= range_time[1])

    ground_spliced = ground[ground_splicer].copy()
    meas_spliced = meas[meas_splicer].copy()

    ground_spliced['time'] = ground_spliced['time'] - range_time[0]
    meas_spliced['time'] = meas_spliced['time'] - range_time[0]

    return meas_spliced, ground_spliced

def pare_data(dataset): #Rework data so that it is all nicely in the DF
    ground = load_ground_truth(f'data/ds{dataset}_Groundtruth.dat')
    meas = load_measurements_data(f'data/ds{dataset}_Measurement.dat')
    meas_rebased, ground_rebased = time_splice(meas, ground)

    landmark_grounds = load_landmark_pos(f'data/ds{dataset}_Landmark_Groundtruth.dat')
    landmark_bar = load_barcodes(f'data/ds{dataset}_Barcodes.dat')
    landmark_grounds['barcode'] = landmark_bar['barcode']
    landmark_dict = {row['barcode']: {'pos': (row['x [m]'], row['y [m]'])} for index, row in landmark_grounds.iterrows()}
    
    subjects = landmark_grounds['barcode'].tolist()
    meas_filtered = meas_rebased[meas_rebased['subject'].isin(subjects)].copy()
    
    meas_filtered['land_x'] = meas_filtered['subject'].map(lambda x: landmark_dict[x]['pos'][0])
    meas_filtered['land_y'] = meas_filtered['subject'].map(lambda x: landmark_dict[x]['pos'][1])
    
    meas_filtered['sin_bearing'] = np.sin(meas_filtered['bearing'])
    meas_filtered['cos_bearing'] = np.cos(meas_filtered['bearing'])

    big_data = data_craft(meas_filtered, ground_rebased)

    big_data = add_temporal_features(big_data, window_size=3)
    
    return big_data, ground_rebased, landmark_dict


def data_craft(filtered_meas, ground_rebase): #Add to the DF the outputs
    x_interp = interp.interp1d(ground_rebase['time'], ground_rebase['x_pos'])
    y_interp = interp.interp1d(ground_rebase['time'], ground_rebase['y_pos'])
    theta_interp = interp.interp1d(ground_rebase['time'], ground_rebase['angle'])

    big_data = filtered_meas.iloc[1:-1].copy()
    big_data['x_pos'] = x_interp(filtered_meas['time'][1:-1])
    big_data['y_pos'] = y_interp(filtered_meas['time'][1:-1])
    big_data['angle'] = theta_interp(filtered_meas['time'][1:-1])
    
    big_data['sin_angle'] = np.sin(big_data['angle'])
    big_data['cos_angle'] = np.cos(big_data['angle'])

    return big_data

def split(data): #Split data into training and test
    shuffle = data.sample(frac = 1, random_state = 2)
    ratio = .8
    total_rows = shuffle.shape[0]
    train_size = int(total_rows*ratio)

    train = shuffle[0:train_size]
    test = shuffle[train_size:]

    return train, test

def gen_data(n_samples = 1000, noise = .20, seed = 42): #Generate the synthetic data
    rng = np.random.default_rng(seed)

    x = np.linspace(0, 4*np.pi, n_samples)

    in_1 = x + rng.normal(0, noise, n_samples)
    in_2 = x**2 / 10 + rng.normal(0, noise * 2, n_samples)
    in_3 = np.cos(x/2) + rng.normal(0, noise * .5, n_samples)

    out_1 = np.sin(x)
    out_2 = np.cos(x)

    dat_dic = {'x': x, 'input_1': in_1,'input_2': in_2, 'input_3': in_3, 'output_1': out_1, 'output_2': out_2}

    synth_dat = pd.DataFrame(dat_dic)

    return synth_dat

def conv_dat(data, input_cols, output_cols): #Convert data to arrays
    inputs = data[input_cols].values
    outputs = data[output_cols].values

    big_dat = np.hstack([inputs, outputs])

    return big_dat, inputs, outputs

def add_temporal_features(big_data, window_size=3): #Last minute addition of temporal features.
    for lag in range(1, window_size):
        big_data[f'range_lag{lag}'] = big_data['range'].shift(lag)
        big_data[f'sin_bearing_lag{lag}'] = np.sin(big_data['bearing'].shift(lag))
        big_data[f'cos_bearing_lag{lag}'] = np.cos(big_data['bearing'].shift(lag))
        big_data[f'land_x_lag{lag}'] = big_data['land_x'].shift(lag)
        big_data[f'land_y_lag{lag}'] = big_data['land_y'].shift(lag)
    
    big_data = big_data.dropna().reset_index(drop=True)
    
    return big_data