import numpy as np
from src.motion import wrap_angle
from src.motion import arc_step
import time
from src.astar import online_astar
from src.grid import world_to_grid
from src.grid import grid_to_world
from src.grid import build_rob_grid
from src.astar import find_neighbors
from src.astar import astar

def initial_vals():
    omega_max = 5.579
    vel_max = .288
    rate = .1

    return omega_max, vel_max, rate

def project(point, seg, length): #Projects the position of the robot onto the polyline and finds the arc it lies on
    seg_point = np.array(point) - np.array(seg[0])
    poly = np.array(seg[1]) - np.array(seg[0])

    scale = np.dot(seg_point, poly)/np.dot(poly, poly)
    scale = np.min([1, np.max([0, scale])])

    proj = seg[0] + scale * poly
    arc_proj = length + scale * np.linalg.norm(poly)

    return proj, arc_proj

def polymath(path, look_ahead, pos, seg_i, length): #Is determining the next target from the projection of the robot onto line based on the segments, then calculates velocities
    
    slow_thresh = .75

    seg = [path[seg_i], path[seg_i + 1]]
    
    proj, proj_arc = project(pos[:-1], seg, length[seg_i])

    if proj_arc >= length[seg_i + 1]:
        seg_i += 1
        if seg_i > len(length) - 2:
            seg_i = len(length) - 2
        seg = [path[seg_i], path[seg_i + 1]]         
        proj, proj_arc = project(pos[:-1], seg, length[seg_i])

    targ_seg = proj_arc + look_ahead

    if targ_seg > length[-1]:
        targ = path[-1]

    elif targ_seg <= length[seg_i+1] and targ_seg > length[seg_i]:
        slope = (targ_seg - length[seg_i])/(length[seg_i+1] - length[seg_i])
        targ = ((1-slope)*seg[0][0] + slope*seg[1][0], (1-slope)*seg[0][1] + slope*seg[1][1])
    
    elif targ_seg >= length[seg_i+1]:
        targ_index = next(x[0] for x in enumerate(length) if x[1] >= targ_seg)
        slope = (targ_seg - length[targ_index - 1])/(length[targ_index] - length[targ_index-1])
        targ = ((1-slope)*path[targ_index-1][0] + slope*path[targ_index][0], (1-slope)*path[targ_index-1][1] + slope*path[targ_index][1])

    targ_range = np.sqrt((pos[0] - targ[0])**2 + (pos[1] - targ[1])**2)
    bear_e = wrap_angle(np.arctan2(targ[1] - pos[1], targ[0] - pos[0]) - pos[2])

    arc_corr = 2*np.sin(bear_e)/look_ahead
    vel_des = np.cos(bear_e)**2 * np.min([1, targ_range/slow_thresh])
    omega_des = np.max([vel_des, .3]) * arc_corr

    return targ_range, bear_e, vel_des, omega_des, seg_i


def simulate(path, control, start, look_ahead): #Creates the simulated robot path based on the premade a* path

    omega_max, vel_max, rate = initial_vals()
    length = [0]
    thresh = .1
    seg_i = 0
    pos = start
    rob_path = [start]
    range_e = []
    angle_e = []
    theta_thresh = .5
    bear_e = theta_thresh + 1
    targ_range = thresh + 1
    std_dev_angle = .005
    std_dev_lin = .002

    for i in range(1, len(path)):
        length.append(length[i-1] + np.linalg.norm(np.array(path[i]) - np.array(path[i-1])))

    while np.linalg.norm(np.array(pos[:-1]) - np.array(path[-1])) > thresh and seg_i <= len(length) - 2: # or np.abs(bear_e) > theta_thresh
        angle_noise = np.random.normal(0, std_dev_angle) 
        lin_noise = np.random.normal(0, std_dev_lin)
        targ_range, bear_e, vel_des, omega_des, seg_i = polymath(path, look_ahead, pos, seg_i, length)

        del_vel = np.clip(vel_des - control[0], -vel_max * rate, vel_max * rate)
        del_omega = np.clip(omega_des - control[1], -omega_max * rate, omega_max * rate)

        new_control = (del_vel + control[0] + lin_noise, del_omega + control[1] + angle_noise)
        control = new_control
        range_e.append(targ_range)
        angle_e.append(bear_e)
        pos = arc_step(pos, control, rate)
        rob_path.append(pos)

    return rob_path, range_e, angle_e

def online_drive(start, goal, control, look_ahead, grid, res, x_range, y_range, heuristic): #Is the online drive. Essentially online a* but with controls
    pos = start  + (-np.pi/2,)
    thresh = .1
    goal_grid = world_to_grid(goal, res, x_range, y_range)
    rob_path = [pos]
    std_dev_angle = .005
    std_dev_lin = .002
    omega_max, vel_max, rate = initial_vals()
    range_e = []
    angle_e = []
    updated_grid = build_rob_grid(grid)
    start_grid = world_to_grid(start, res, x_range, y_range) 
    updated_grid[start_grid[0], start_grid[1]] = 0
    nn = find_neighbors(start_grid, updated_grid)
    for i in nn:
        updated_grid[i[0], i[1]] = grid[i[0], i[1]]
    plan_grid_path = [start_grid]
    seg_i = 0
    need_plan = True
    current_grid = world_to_grid(pos[:-1], res, x_range, y_range)

    while np.linalg.norm(np.array(pos[:-1]) - np.array(goal)) > thresh:
        
        if need_plan:
            start_plan = grid_to_world(current_grid, res, x_range, y_range)
            path_list, expansions = astar(start_plan, goal, updated_grid, res, x_range, y_range, heuristic)
            path_world = []
            for x in path_list:
                path_world.append(grid_to_world(x, res, x_range, y_range))
            
            length = [0]
            for i in range(1, len(path_world)):
                length.append(length[i-1] + np.linalg.norm(np.array(path_world[i]) - np.array(path_world[i-1])))
            seg_i = 0

            need_plan = False

        if path_list:
            
            angle_noise = np.random.normal(0, std_dev_angle) 
            lin_noise = np.random.normal(0, std_dev_lin)
            targ_range, bear_e, vel_des, omega_des, seg_i = polymath(path_world, look_ahead, pos, seg_i, length)

            del_vel = np.clip(vel_des - control[0], -vel_max * rate, vel_max * rate)
            del_omega = np.clip(omega_des - control[1], -omega_max * rate, omega_max * rate)

            new_control = (del_vel + control[0] + lin_noise, del_omega + control[1] + angle_noise)
            control = new_control
            pos = arc_step(pos, control, rate)

            current_grid = world_to_grid(pos[:-1], res, x_range, y_range)

            nn = find_neighbors(current_grid, updated_grid)

            for i in nn: #update with known values
                updated_grid[i[0], i[1]] = grid[i[0], i[1]]
                if updated_grid[i[0], i[1]] == 1 and i in path_list:
                    need_plan = True
    
            rob_path.append(pos)
            plan_grid_path.append(current_grid)
            range_e.append(targ_range)
            angle_e.append(bear_e)
            
        else:
            break
    
    return rob_path, range_e, angle_e

