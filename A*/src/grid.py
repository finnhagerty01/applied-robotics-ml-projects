import pandas as pd
import numpy as np
import itertools

def load_landmarks():
    landmarks = pd.read_csv('data/ds1_Landmark_Groundtruth.dat', sep = '\s+', skiprows = 4, header= None, names = ['x_pos', 'y_pos', 'x_dev', 'y_dev'])
    return landmarks

def world_to_grid(pos, res, x_range, y_range) -> tuple: #Change from world position to grid index
    index_y = int(np.floor((pos[1]-y_range[0])/res))
    index_x = int(np.floor((pos[0]-x_range[0])/res))
    grid_index = (index_y, index_x)
    
    if pos[1] >= y_range[1]: #Consider edge cases
        grid_index = (index_y - 1, index_x)
    if pos[0] >= x_range[1]:
        grid_index = (index_y, index_x - 1)
    return grid_index

def grid_to_world(index, res, x_range, y_range) -> tuple: #transfer from grid to world for plotting
    x_w = x_range[0] + (index[1] + .5) * res
    y_w = y_range[0] + (index[0] + .5) * res

    return (x_w, y_w)

def build_grids(res, x_range, y_range) -> np.array: #build the grid
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]

    landmark = load_landmarks()
    landmark_pos = landmark[['x_pos', 'y_pos']]

    grid = np.zeros([int(y_size/res), int(x_size/res)]) #initial grid w/ zeros for unoccupied

    for index, row in landmark_pos.iterrows(): #Occupy grid with landmarks
        landmark_grid = world_to_grid((row['x_pos'], row['y_pos']), res, x_range, y_range)
        grid[landmark_grid] = 1

    return grid

def build_fine_grid(res, x_range, y_range) -> np.array:
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]

    landmark = load_landmarks()
    landmark_pos = landmark[['x_pos', 'y_pos']]

    grid = np.zeros([int(y_size/res), int(x_size/res)]) #initial grid w/ zeros for unoccupied
    rows = grid.shape[0]
    cols = grid.shape[1]

    for index, row in landmark_pos.iterrows(): #Occupy grid with landmarks
        landmark_grid = world_to_grid((row['x_pos'], row['y_pos']), res, x_range, y_range)

        y_loc, x_loc = landmark_grid

        y_ind_n = [y_loc, y_loc + 1, y_loc - 1, y_loc + 2, y_loc - 2, y_loc + 3, y_loc - 3]
        x_ind_n = [x_loc, x_loc + 1, x_loc -1, x_loc +2, x_loc -2, x_loc +3, x_loc -3]
        
        expansion = list(itertools.product(y_ind_n, x_ind_n))

        for i in expansion.copy():
            if i[0] >= rows or i[0] < 0 or i[1] >= cols or i[1] < 0:
                expansion.remove(i)

        for i in expansion:
            grid[i] = 1

    return grid

def build_rob_grid(grid):
    return np.full((grid.shape[0], grid.shape[1]), -1, dtype = int)

if __name__ == '__main__': 
    build_grids(1, [-2, 5], [-6, 6])