from src.astar import astar, online_astar
from src.grid import build_grids, build_fine_grid
from src.initialize import initialize
from src.initialize import load_sg3, load_sg7
from src.visualize import visualize
from src.control import simulate
from src.grid import grid_to_world
import numpy as np
from src.visualize import visualize_rob
from src.control import online_drive
from src.visualize import visualize_grid
from src.heuristics import euc_heur, heur

if __name__ == "__main__":
    x_range, y_range, start, goal = initialize(load_sg3())
    res = 1
    grid = build_grids(res, x_range, y_range)
    j = 0

    visualize_grid(grid, x_range, y_range, res, f'plots/justgrid_coarse')
    for i in range(len(start)):
        path_list, expansions = astar(start[i], goal[i], grid, res, x_range, y_range, heur)
        visualize(path_list, grid, x_range, y_range, res, f'plots/step3_plot_case_{j}_cheb')
        path_list, expansions = astar(start[i], goal[i], grid, res, x_range, y_range, euc_heur)
        visualize(path_list, grid, x_range, y_range, res, f'plots/step3_plot_case_{j}_euc')
        j += 1

    for i in range(len(start)):
        path_list, expansions, rob_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, heur)
        visualize(path_list, grid, x_range, y_range, res, f'plots/step5_plot_case_{j}_cheb')
        path_list, expansions, rob_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, euc_heur)
        visualize(path_list, grid, x_range, y_range, res, f'plots/step5_plot_case_{j}_euc')
        j += 1

    x_range, y_range, start, goal = initialize(load_sg7())
    res = .1
    grid = build_fine_grid(res, x_range, y_range)
    j = 0
    visualize_grid(grid, x_range, y_range, res, f'plots/justgrid_fine')

    for i in range(len(start)):
        path_list, expansions, rob_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, heur)
        visualize(path_list, grid, x_range, y_range, res, f'plots/step7_plots{j}_cheb')
        path_list, expansions, rob_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, euc_heur)
        visualize(path_list, grid, x_range, y_range, res, f'plots/step7_plots{j}_euc')
        j += 1
    
    j = 0

    for i in range(len(start)):
        path_list, expansions, update_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, heur)
        path_world = []
        for x in path_list:
            path_world.append(grid_to_world(x, res, x_range, y_range))
        rob_path, range_e, angle_e = simulate(path_world, (0, 0), start[i] + (-np.pi/2,), .1)
        visualize_rob(rob_path, path_list, grid, x_range, y_range, res, f'plots/step9_plot{j}_cheb', j)

        path_list, expansions, update_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, euc_heur)
        path_world = []
        for x in path_list:
            path_world.append(grid_to_world(x, res, x_range, y_range))
        rob_path, range_e, angle_e = simulate(path_world, (0, 0), start[i] + (-np.pi/2,), .1)
        visualize_rob(rob_path, path_list, grid, x_range, y_range, res, f'plots/step9_plot{j}_euc', j)
        j += 1

    j = 0

    for i in range(len(start)):
        
        rob_path, range_e, angle_e = online_drive(start[i], goal[i], (0, 0), .1, grid, res, x_range, y_range, heur)
        path_list, expansions, rob_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, heur)
        visualize_rob(rob_path, path_list, grid, x_range, y_range, res, f'plots/step10_plot{j}_cheb', j)

        rob_path, range_e, angle_e = online_drive(start[i], goal[i], (0, 0), .1, grid, res, x_range, y_range, euc_heur)
        path_list, expansions, rob_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, euc_heur)
        visualize_rob(rob_path, path_list, grid, x_range, y_range, res, f'plots/step10_plot{j}_euc', j)
        j += 1

    x_range, y_range, start, goal = initialize(load_sg3())
    res = .1
    grid = build_fine_grid(res, x_range, y_range)
    j = 0

    for i in range(len(start)):
        rob_path, range_e, angle_e = online_drive(start[i], goal[i], (0, 0), .1, grid, res, x_range, y_range, heur)
        path_list, expansions, rob_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, heur)
        visualize_rob(rob_path, path_list, grid, x_range, y_range, res, f'plots/step11_plot{j}_cheb', j)

        rob_path, range_e, angle_e = online_drive(start[i], goal[i], (0, 0), .1, grid, res, x_range, y_range, euc_heur)
        path_list, expansions, rob_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, euc_heur)
        visualize_rob(rob_path, path_list, grid, x_range, y_range, res, f'plots/step11_plot{j}_euc', j)
        j += 1

    x_range, y_range, start, goal = initialize(load_sg3())
    res = 1
    grid = build_grids(res, x_range, y_range)
    j = 0
    for i in range(len(start)):
        rob_path, range_e, angle_e = online_drive(start[i], goal[i], (0, 0), .1, grid, res, x_range, y_range, heur)
        path_list, expansions, rob_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, heur)
        visualize_rob(rob_path, path_list, grid, x_range, y_range, res, f'plots/step11_coarse_plot{j}_cheb', j)
        
        rob_path, range_e, angle_e = online_drive(start[i], goal[i], (0, 0), .1, grid, res, x_range, y_range, euc_heur)
        path_list, expansions, rob_grid = online_astar(start[i], goal[i], grid, res, x_range, y_range, euc_heur)
        visualize_rob(rob_path, path_list, grid, x_range, y_range, res, f'plots/step11_coarse_plot{j}_euc', j)
        j += 1