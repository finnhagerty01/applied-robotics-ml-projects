import numpy as np
import pandas as pd
import itertools
from src.heuristics import heur, euc_heur
from src.grid import world_to_grid, build_rob_grid, grid_to_world
import heapq
import time

def find_neighbors(loc, grid) -> list: #Find nearest neighbors (nn)
    rows = grid.shape[0]
    cols = grid.shape[1]

    y_loc, x_loc = loc

    y_ind_n = [y_loc, y_loc + 1, y_loc - 1]
    x_ind_n = [x_loc, x_loc + 1, x_loc -1]
    
    nn = list(itertools.product(y_ind_n, x_ind_n)) #cool nifty tool to get all combos of neighbors
    nn.remove((y_loc, x_loc)) #get rid of position of bot

    for i in nn.copy():
        if i[0] >= rows or i[0] < 0 or i[1] >= cols or i[1] < 0:
            nn.remove(i)

    # for i in nn.copy(): #Get outta here if you are occupied
    #     if grid[i[0], i[1]] == 1:
    #         nn.remove(i)
    
    return nn

def astar(start, goal, grid, res, x_range, y_range, heuristic) -> list:  
    start_grid = world_to_grid(start, res, x_range, y_range)
    goal_grid = world_to_grid(goal, res, x_range, y_range)
    pos_grid = start_grid

    path = {start_grid: None}
    expansions = {}
    step_cost = 1
    occ_cost = 1000
    unk_cost = 10

    h = heuristic(start_grid, goal_grid)
    g = {pos_grid: 0}
    f = g[pos_grid] + h

    open_set = []
    heapq.heappush(open_set, (f, -g[pos_grid], pos_grid))

    while open_set != []:
        
        f, tie, node = heapq.heappop(open_set)
        
        f_test = g[node] + heuristic(node, goal_grid)
        if f > f_test:
            continue
        
        if node == goal_grid:
            break

        nn = find_neighbors(node, grid)
        expansions[node] = nn
        
        for i in nn:
            h = heuristic(i, goal_grid)
            if grid[i[0], i[1]] == 1:
                test_g = g[node] + occ_cost
            elif grid[i[0], i[1]] == -1:
                test_g = g[node] + unk_cost
            else:
                test_g = g[node] + step_cost

            if i not in g:
                g[i] = test_g
                f = g[i] + h
                path[i] = node
                heapq.heappush(open_set, (f, -g[i], i))
            elif test_g < g[i]:
                g[i] = test_g
                f = g[i] + h
                path[i] = node
                heapq.heappush(open_set, (f, -g[i], i))

        pos_grid = node
    
    if node != goal_grid:
        path_list = []
    
    else:
        path_list = [goal_grid]
        key = goal_grid
        
        while key != None:
            path_list.append(path[key])
            key = path[key]

        path_list.remove(None)
        path_list.reverse()
    
    return path_list, expansions

def online_astar(start, goal, grid, res, x_range, y_range, heuristic, rob_grid = None) -> list:
    start_grid = world_to_grid(start, res, x_range, y_range)
    goal_grid = world_to_grid(goal, res, x_range, y_range)
    pos_grid = start_grid
    
    if rob_grid is None:
        rob_grid = build_rob_grid(grid)
    nn = find_neighbors(start_grid, grid)
    rob_grid[start_grid[0], start_grid[1]] = 0

    for i in nn: #update starting position with known values
        rob_grid[i[0], i[1]] = grid[i[0], i[1]]

    path = {start_grid: None}
    expansions = {}
    need_plan = True
    pred_path = []

    while pos_grid != goal_grid:
        nn = find_neighbors(pos_grid, rob_grid)

        for i in nn: #update with known values
            rob_grid[i[0], i[1]] = grid[i[0], i[1]]
        
            if rob_grid[i[0], i[1]] == 1 and i in pred_path:
                need_plan = True
                pred_path = []
        
        if need_plan == True or not pred_path:
            pred_path, pred_exp = astar(grid_to_world(pos_grid, res, x_range, y_range), goal, rob_grid,res, x_range, y_range, heuristic)
            if pred_path[0] == pos_grid:
                pred_path = pred_path[1:]
            need_plan = False
        
        if not pred_path:
            break
            
        parent = pos_grid
        pos_grid = pred_path.pop(0)

        path[pos_grid] = parent
        
    if pos_grid != goal_grid:
        path_list = []
    
    else:
        path_list = [goal_grid]
        key = goal_grid

        while key != None:
            path_list.append(path[key])
            key = path[key]

        path_list.remove(None)
        path_list.reverse()

    return path_list, expansions, rob_grid



