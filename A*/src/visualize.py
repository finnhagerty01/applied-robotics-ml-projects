import numpy as np
import matplotlib.pyplot as plt
from src.grid import grid_to_world

def visualize(path, grid, x_range, y_range, res, output):

    world_path = []
    for i in path:
        world_path.append(grid_to_world(i, res, x_range, y_range))

    x_path, y_path = zip(*world_path)

    plt.figure()
    plt.imshow(grid, origin = 'lower', cmap = 'gray_r', extent = [x_range[0], x_range[1], y_range[0], y_range[1]])
    plt.plot(x_path, y_path, '-r')
    plt.scatter(x_path[0], y_path[0], color = 'g', marker = 'o', s = 10)
    plt.scatter(x_path[-1], y_path[-1], color = 'r', marker = 'o', s = 10)
    for x in np.arange(x_range[0], x_range[1]+res, res):
        plt.vlines(x, y_range[0], y_range[1], color='gray', alpha=0.2, lw=0.3)
    for y in np.arange(y_range[0], y_range[1]+res, res):
        plt.hlines(y, x_range[0], x_range[1], color='gray', alpha=0.2, lw=0.3)
    plt.title('$A^*$ path of robot')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True)
    plt.xticks(range(x_range[0], x_range[1] + 1, 1))
    plt.yticks(range(y_range[0], y_range[1] + 1, 1))
    plt.legend(['robot path', 'start', 'end'], loc = 'lower right')
    
    plt.savefig(output)
    plt.show()
    
def visualize_rob(rob_path, a_path, grid, x_range, y_range, res, output, j):
    
    world_path = []
    for i in a_path:
        world_path.append(grid_to_world(i, res, x_range, y_range))

    xa_path, ya_path = zip(*world_path)
    xr_path, yr_path, thetas = zip(*rob_path)

    plt.figure()
    plt.imshow(grid, origin = 'lower', cmap = 'gray_r', extent = [x_range[0], x_range[1], y_range[0], y_range[1]])
    plt.plot(xa_path, ya_path, '-r')
    plt.scatter(xa_path[0], ya_path[0], color = 'g', marker = 'o', s = 10)
    plt.scatter(xa_path[-1], ya_path[-1], color = 'r', marker = 'o', s = 10)
    plt.plot(xr_path, yr_path, '-k')
    plt.quiver(xr_path[::20], yr_path[::20], 1, 0, pivot = 'tail', angles = np.degrees(thetas[::20]), scale = 30, color = 'c', zorder = 2)
    for x in np.arange(x_range[0], x_range[1]+res, res):
        plt.vlines(x, y_range[0], y_range[1], color='gray', alpha=0.2, lw=0.3)
    for y in np.arange(y_range[0], y_range[1]+res, res):
        plt.hlines(y, x_range[0], x_range[1], color='gray', alpha=0.2, lw=0.3)
    plt.title('$A^*$ path of robot')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True)
    plt.xticks(range(x_range[0], x_range[1] + 1, 1))
    plt.yticks(range(y_range[0], y_range[1] + 1, 1))
    if j < 2:
        plt.legend(['planned path', 'start', 'end', 'robot path'], loc = 'upper right')
    else:
        plt.legend(['planned path', 'start', 'end', 'robot path'], loc = 'lower right')
    
    plt.savefig(output)
    plt.show()

def visualize_grid(grid, x_range, y_range, res, output):
    plt.figure()
    plt.imshow(grid, origin = 'lower', cmap = 'gray_r', extent = [x_range[0], x_range[1], y_range[0], y_range[1]])
    plt.title('$A^*$ path of robot')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True)
    for x in np.arange(x_range[0], x_range[1]+res, res):
        plt.vlines(x, y_range[0], y_range[1], color='gray', alpha=0.2, lw=0.3)
    for y in np.arange(y_range[0], y_range[1]+res, res):
        plt.hlines(y, x_range[0], x_range[1], color='gray', alpha=0.2, lw=0.3)
    plt.xticks(range(x_range[0], x_range[1] + 1, 1))
    plt.yticks(range(y_range[0], y_range[1] + 1, 1))
    plt.savefig(output)
    plt.show()


