import numpy as np
import pandas as pd

#Simply just standardizes all my data, and willl also destandardize for visualization
def standardize_fit(data, stand_ids, pass_ids):

    stats = {'stand_ids': stand_ids, 'pass_ids': pass_ids}

    mean = data.mean(axis = 0)
    scales = data.std(axis = 0, ddof = 0)
    scales = np.clip(scales, 1e-12, None)

    mean[pass_ids] = 0
    scales[pass_ids] = 1

    stats['means'] = mean
    stats['scales'] = scales

    return stats

def standardize_transform(data, stats):
    
    stand_ids = stats['stand_ids']
    means = stats['means']
    scales = stats['scales']

    normal_data = data.copy()
    normal_data[:, stand_ids] = (data[:, stand_ids] - means[stand_ids])/scales[stand_ids]


    return normal_data

def inverse_transform(data, stats):
    new_data = data.copy()
    stand_ids = stats['stand_ids']
    means = stats['means']
    scales = stats['scales']
    
    new_data[:, stand_ids] = new_data[:, stand_ids] * scales[stand_ids] + means[stand_ids]

    return new_data

def stand_pipeline(data, stand_ids, pass_ids):
    stats = standardize_fit(data, stand_ids, pass_ids)
    new_dat = standardize_transform(data, stats)

    return new_dat, stats
